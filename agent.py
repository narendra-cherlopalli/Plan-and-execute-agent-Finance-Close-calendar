"""
plan_execute_agent/agent.py — Plan-and-execute pattern.

Domain: Financial close calendar planning.

The agent first produces a COMPLETE multi-step plan (task list with
dependencies and owners) before executing a single task. This is the
defining trait: planning is a distinct phase, finished before execution
begins — unlike ReAct, which interleaves reasoning and acting one step
at a time with no upfront commitment to a full sequence.

The agent also handles REPLANNING: if a task slips (misses its target
date), the agent doesn't just mark it late — it recomputes the plan,
identifying which downstream tasks are now at risk and whether the
overall close deadline is still achievable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class Task:
    task_id: str
    name: str
    owner: str
    duration_days: int
    depends_on: list[str] = field(default_factory=list)
    planned_start: Optional[date] = None
    planned_end: Optional[date] = None
    status: str = "NOT_STARTED"  # NOT_STARTED | IN_PROGRESS | COMPLETE | BLOCKED
    actual_end: Optional[date] = None


@dataclass
class ClosePlan:
    tasks: dict[str, Task]
    target_close_date: date
    is_achievable: bool
    critical_path: list[str] = field(default_factory=list)


@dataclass
class ReplanResult:
    plan: ClosePlan
    at_risk_tasks: list[str]
    deadline_still_achievable: bool
    days_slipped: int


class CloseCalendarAgent:
    """
    Plan-and-execute agent for close calendar scheduling.

    PLAN phase: given a task list with dependencies, compute the full
    schedule via topological sort + critical path analysis, BEFORE any
    task starts.

    EXECUTE phase (here represented as record_actual_completion): record
    real task completions. If a task slips, trigger REPLAN — not just a
    status update, but a full recomputation of downstream impact.
    """

    def __init__(self, target_close_date: date) -> None:
        self.target_close_date = target_close_date

    # ─────────────────────────────────────────────────────────────────────────
    # PLAN — full upfront planning, before any execution
    # ─────────────────────────────────────────────────────────────────────────

    def plan(self, task_defs: list[dict], start_date: date) -> ClosePlan:
        tasks = {
            t["task_id"]: Task(
                task_id=t["task_id"],
                name=t["name"],
                owner=t["owner"],
                duration_days=t["duration_days"],
                depends_on=t.get("depends_on", []),
            )
            for t in task_defs
        }

        ordered = self._topological_sort(tasks)
        self._schedule(tasks, ordered, start_date)

        critical_path = self._compute_critical_path(tasks, ordered)
        final_end_dates = [t.planned_end for t in tasks.values() if t.planned_end]
        plan_end = max(final_end_dates) if final_end_dates else start_date
        is_achievable = plan_end <= self.target_close_date

        return ClosePlan(
            tasks=tasks,
            target_close_date=self.target_close_date,
            is_achievable=is_achievable,
            critical_path=critical_path,
        )

    def _topological_sort(self, tasks: dict[str, Task]) -> list[str]:
        """Standard Kahn's algorithm — order tasks so dependencies always precede dependents."""
        in_degree = {tid: 0 for tid in tasks}
        for t in tasks.values():
            for dep in t.depends_on:
                if dep in in_degree:
                    in_degree[t.task_id] += 1

        queue = [tid for tid, deg in in_degree.items() if deg == 0]
        ordered: list[str] = []

        while queue:
            queue.sort()  # deterministic ordering among ties
            current = queue.pop(0)
            ordered.append(current)
            for t in tasks.values():
                if current in t.depends_on:
                    in_degree[t.task_id] -= 1
                    if in_degree[t.task_id] == 0:
                        queue.append(t.task_id)

        if len(ordered) != len(tasks):
            unresolved = set(tasks.keys()) - set(ordered)
            raise ValueError(f"Circular dependency detected involving: {unresolved}")

        return ordered

    def _schedule(self, tasks: dict[str, Task], ordered: list[str], start_date: date) -> None:
        """Assign planned_start/planned_end to each task in topological order."""
        for tid in ordered:
            t = tasks[tid]
            if not t.depends_on:
                t.planned_start = start_date
            else:
                dep_ends = [tasks[d].planned_end for d in t.depends_on if d in tasks and tasks[d].planned_end]
                t.planned_start = max(dep_ends) if dep_ends else start_date
            t.planned_end = self._add_business_days(t.planned_start, t.duration_days)

    def _compute_critical_path(self, tasks: dict[str, Task], ordered: list[str]) -> list[str]:
        """The chain of tasks whose end date equals the overall plan end date."""
        if not tasks:
            return []
        plan_end = max(t.planned_end for t in tasks.values() if t.planned_end)
        path: list[str] = []
        current_end = plan_end
        # Walk backward from whichever task ends last, following the
        # dependency that determined its start date.
        candidates = [t for t in tasks.values() if t.planned_end == current_end]
        node = candidates[0] if candidates else None

        while node is not None:
            path.insert(0, node.task_id)
            if not node.depends_on:
                break
            deps_with_matching_end = [
                tasks[d] for d in node.depends_on
                if d in tasks and tasks[d].planned_end == node.planned_start
            ]
            node = deps_with_matching_end[0] if deps_with_matching_end else None

        return path

    # ─────────────────────────────────────────────────────────────────────────
    # EXECUTE + REPLAN — record reality, recompute downstream impact
    # ─────────────────────────────────────────────────────────────────────────

    def record_actual_completion(
        self, plan: ClosePlan, task_id: str, actual_end_date: date
    ) -> ReplanResult:
        """
        Record that a task finished on actual_end_date. If it slipped past
        its planned_end, trigger a full replan: recompute every downstream
        task's schedule and re-check overall deadline achievability.
        """
        task = plan.tasks[task_id]
        task.actual_end = actual_end_date
        task.status = "COMPLETE"

        slip_days = max(0, (actual_end_date - task.planned_end).days) if task.planned_end else 0

        if slip_days == 0:
            return ReplanResult(
                plan=plan,
                at_risk_tasks=[],
                deadline_still_achievable=plan.is_achievable,
                days_slipped=0,
            )

        # Task slipped — replan downstream tasks from the new actual end date.
        at_risk = self._find_downstream(plan.tasks, task_id)
        self._reschedule_downstream(plan.tasks, task_id, actual_end_date)

        final_end_dates = [t.planned_end for t in plan.tasks.values() if t.planned_end]
        new_plan_end = max(final_end_dates) if final_end_dates else actual_end_date
        deadline_achievable = new_plan_end <= self.target_close_date
        plan.is_achievable = deadline_achievable

        return ReplanResult(
            plan=plan,
            at_risk_tasks=at_risk,
            deadline_still_achievable=deadline_achievable,
            days_slipped=slip_days,
        )

    def _find_downstream(self, tasks: dict[str, Task], task_id: str) -> list[str]:
        """All tasks that directly or transitively depend on task_id."""
        downstream: set[str] = set()
        changed = True
        while changed:
            changed = False
            for t in tasks.values():
                if t.task_id in downstream:
                    continue
                if task_id in t.depends_on or any(d in downstream for d in t.depends_on):
                    downstream.add(t.task_id)
                    changed = True
        return sorted(downstream)

    def _reschedule_downstream(self, tasks: dict[str, Task], slipped_task_id: str, new_end: date) -> None:
        """Recompute planned_start/end for every task downstream of the slipped task."""
        tasks[slipped_task_id].planned_end = new_end
        ordered = self._topological_sort(tasks)
        for tid in ordered:
            t = tasks[tid]
            if t.task_id == slipped_task_id or not t.depends_on:
                continue
            dep_ends = [tasks[d].planned_end for d in t.depends_on if d in tasks and tasks[d].planned_end]
            if dep_ends:
                new_start = max(dep_ends)
                if t.status != "COMPLETE":
                    t.planned_start = new_start
                    t.planned_end = self._add_business_days(new_start, t.duration_days)

    @staticmethod
    def _add_business_days(start: date, days: int) -> date:
        """Add N business days (skip weekends) to a start date."""
        current = start
        added = 0
        while added < days:
            current += timedelta(days=1)
            if current.weekday() < 5:  # Mon-Fri
                added += 1
        return current
