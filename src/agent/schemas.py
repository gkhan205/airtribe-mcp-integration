"""
schemas.py

Pydantic models for STRUCTURED OUTPUT.

Instead of asking Claude for free text and parsing it, we do:

    structured_llm = get_llm().with_structured_output(JobRequirements)
    reqs = structured_llm.invoke("...job description...")
    reqs.must_have   # -> a real Python list

Claude reads the field descriptions below, so they are part of the prompt.
"""

from typing import Literal

from pydantic import BaseModel, Field


class JobRequirements(BaseModel):
    """What a job description (or a recruiter's search request) asks for."""

    title: str = Field(description="The job title, e.g. 'IT Systems Administrator'. Make one up if only skills are given.")
    must_have: list[str] = Field(description="Skills or requirements the candidate MUST have")
    nice_to_have: list[str] = Field(description="Skills that are a bonus but not required")
    min_years: int = Field(description="Minimum years of overall experience asked for, 0 if not stated")
    education: str | None = Field(default=None, description="Required degree, if any, e.g. 'Bachelor in Accounting'")


class CandidateAnalysis(BaseModel):
    """Round 2: a deep look at one candidate against the job."""

    summary: str = Field(description="2-3 sentence summary of the candidate's fit for THIS job")
    strengths: list[str] = Field(description="3-5 concrete strengths for this job, with evidence from the resume")
    gaps: list[str] = Field(description="Missing skills, weak areas or risks for this job")
    fit_score: int = Field(description="Overall fit from 1 (poor) to 10 (excellent)")


class HireDecision(BaseModel):
    """Round 3: the final recommendation for one candidate."""

    candidate_id: str = Field(description="The candidate id exactly as given, e.g. 'chen_wang'")
    decision: Literal["hire", "borderline", "no-hire"] = Field(
        description="'hire' = invite to interview, 'borderline' = close but has fixable gaps, 'no-hire' = not a fit"
    )
    reason: str = Field(description="One or two sentences explaining the decision")
    improvement_suggestions: list[str] = Field(
        default_factory=list,
        description="For borderline candidates: what would make them a hire (skills to show, experience to verify). Empty otherwise.",
    )


class FinalDecisions(BaseModel):
    """All round 3 decisions, made together so candidates are judged against each other."""

    decisions: list[HireDecision]


class FeedbackIntent(BaseModel):
    """What the recruiter wants after seeing the results."""

    intent: Literal["refine", "new_job", "question", "done"] = Field(
        description=(
            "'refine' = change the requirements for the SAME job (add/remove a must-have, minimum years); "
            "'new_job' = search for a different job or a brand new set of skills; "
            "'question' = anything else: compare, explain a ranking, interview questions, details about a candidate; "
            "'done' = the user is finished (bye, thanks that's all, exit)"
        )
    )
    add_must: list[str] = Field(
        default_factory=list,
        description="For 'refine': new must-haves, formatted 'Skill' or 'Skill:years' (e.g. 'Linux', 'Python:3') or 'years:N' for overall experience",
    )
    remove_must: list[str] = Field(default_factory=list, description="For 'refine': must-haves to drop, same format")
    new_job_description: str | None = Field(
        default=None, description="For 'new_job': the new job description or search request, in the user's words"
    )


class CandidateComparison(BaseModel):
    """Head-to-head comparison of several candidates."""

    winner_id: str = Field(description="Id of the strongest candidate for this job")
    verdict: str = Field(description="2-3 sentences on who is best and why")
    key_differences: list[str] = Field(description="The most important differences between the candidates")


class InterviewQuestions(BaseModel):
    """Screening questions for one candidate."""

    questions: list[str] = Field(description="5 screening questions; at least 2 must probe the candidate's gaps")
