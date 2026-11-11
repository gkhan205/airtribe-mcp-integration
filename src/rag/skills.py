"""Curated skill taxonomy shared by the indexer (resume_rag.py) and matcher (job_matcher.py).

Each canonical skill maps to regex alias patterns (matched case-insensitively on word
boundaries). Keeping one taxonomy for both sides means "MS Excel" in a resume and
"Excel" in a job description normalize to the same skill.
"""
from __future__ import annotations

import re
from functools import lru_cache

SKILLS: dict[str, dict[str, list[str]]] = {
    "Technology": {
        "Python": [r"python"],
        "Java": [r"java(?!script)"],
        "JavaScript": [r"javascript", r"js"],
        "React": [r"react(?:\.?js)?"],
        "SQL": [r"sql", r"t-sql", r"pl/sql", r"mysql", r"postgres(?:ql)?"],
        "C#": [r"c#", r"\.net"],
        "C++": [r"c\+\+"],
        "HTML/CSS": [r"html5?", r"css3?"],
        "Machine Learning": [r"machine learning", r"ml", r"deep learning"],
        "Data Analysis": [r"data analy(?:sis|tics|st)", r"analytics"],
        "Tableau": [r"tableau"],
        "Power BI": [r"power ?bi"],
        "Linux": [r"linux", r"ubuntu", r"unix", r"red ?hat"],
        "Windows Server": [r"windows server", r"server 20(?:03|08|12|16)"],
        "Active Directory": [r"active directory", r"ad ds"],
        "Networking": [r"networking", r"tcp/ip", r"lan", r"wan", r"vpn", r"dns", r"dhcp", r"cisco", r"routers?"],
        "Virtualization": [r"vmware", r"virtuali[sz]ation", r"hyper-v", r"citrix"],
        "Cloud": [r"aws", r"azure", r"cloud", r"office 365", r"gcp"],
        "Cybersecurity": [r"cyber ?security", r"firewalls?", r"information security", r"network security"],
        "Technical Support": [r"technical support", r"help ?desk", r"troubleshoot(?:ing)?", r"service desk"],
        "ITIL": [r"itil"],
        "SharePoint": [r"sharepoint"],
        "Quality Assurance": [r"quality assurance", r"qa", r"software testing", r"test cases?"],
        "Agile": [r"agile", r"scrum", r"jira"],
        "SAP": [r"sap"],
        "Oracle": [r"oracle"],
        "Salesforce": [r"salesforce"],
        "CRM": [r"crm"],
        "Microsoft Office": [r"microsoft office", r"ms office", r"office suite"],
        "Excel": [r"excel", r"spreadsheets?", r"pivot tables?", r"vlookup"],
        "Microsoft Access": [r"ms access", r"microsoft access"],
    },
    "Engineering": {
        "AutoCAD": [r"autocad", r"cad"],
        "SolidWorks": [r"solidworks", r"solid works"],
        "MATLAB": [r"matlab"],
        "Mechanical Engineering": [r"mechanical engineering", r"mechanical design"],
        "Lean Six Sigma": [r"six sigma", r"lean manufacturing", r"lean", r"kaizen", r"5s"],
        "Quality Control": [r"quality control", r"iso 9001", r"iso", r"root cause analysis", r"capa"],
        "Manufacturing": [r"manufacturing", r"production"],
        "Process Improvement": [r"process improvements?", r"continuous improvement"],
        "Project Management": [r"project management", r"project manager", r"pmp", r"ms project"],
    },
    "Business & Finance": {
        "Accounting": [r"accounting", r"bookkeeping"],
        "Accounts Payable": [r"accounts payable", r"a/p"],
        "Accounts Receivable": [r"accounts receivable", r"a/r", r"collections"],
        "General Ledger": [r"general ledger", r"g/l", r"journal entries"],
        "Reconciliation": [r"reconciliations?", r"reconcil(?:e|ed|ing)"],
        "Financial Reporting": [r"financial reporting", r"financial statements?", r"month-end close", r"month end close", r"gaap"],
        "Financial Analysis": [r"financial analysis", r"financial modeling", r"variance analysis", r"forecasting"],
        "Budgeting": [r"budget(?:ing|s)?"],
        "Auditing": [r"audit(?:ing|s)?"],
        "Tax": [r"tax(?:es|ation)?"],
        "Payroll": [r"payroll", r"adp"],
        "QuickBooks": [r"quickbooks"],
        "Treasury": [r"treasury", r"cash management", r"cash flow"],
        "CPA": [r"cpa", r"certified public accountant"],
        "Banking": [r"banking", r"loans?", r"lending", r"credit"],
        "Compliance": [r"compliance", r"regulatory", r"aml", r"kyc"],
        "Risk Management": [r"risk management", r"risk assessment"],
    },
    "Sales & Customer": {
        "Sales": [r"sales", r"selling"],
        "Business Development": [r"business development", r"lead generation", r"prospecting"],
        "Account Management": [r"account management", r"account manager", r"client relationships?"],
        "Customer Service": [r"customer service", r"customer satisfaction", r"client service"],
        "Negotiation": [r"negotiat(?:ion|ing|ed)"],
        "Retail": [r"retail", r"merchandising", r"point of sale", r"pos", r"cash register"],
        "Marketing": [r"marketing", r"social media", r"seo", r"campaigns?"],
        "Inventory Management": [r"inventory"],
    },
    "HR & People": {
        "Recruiting": [r"recruit(?:ing|ment|er)?", r"talent acquisition", r"staffing", r"interview(?:ing|s)?"],
        "Onboarding": [r"onboarding", r"new hire orientation", r"orientation"],
        "Employee Relations": [r"employee relations", r"conflict resolution", r"grievances?"],
        "Benefits Administration": [r"benefits administration", r"benefits", r"401\(?k\)?", r"fmla"],
        "HRIS": [r"hris", r"workday", r"peoplesoft", r"kronos"],
        "Performance Management": [r"performance management", r"performance reviews?", r"performance appraisals?"],
        "Training & Development": [r"training and development", r"training programs?", r"coaching", r"mentoring"],
        "Labor Law": [r"labor law", r"employment law", r"eeo", r"flsa", r"osha"],
        "Leadership": [r"leadership", r"team lead(?:er)?", r"supervis(?:ed|ion|or|ing)", r"managed a team"],
        "Communication": [r"communication skills", r"verbal communication", r"written communication", r"public speaking"],
    },
    "Design": {
        "Adobe Photoshop": [r"photoshop"],
        "Adobe Illustrator": [r"illustrator"],
        "Adobe InDesign": [r"indesign"],
        "Adobe Creative Suite": [r"adobe creative (?:suite|cloud)", r"adobe cs\d?"],
        "Graphic Design": [r"graphic design(?:er)?", r"visual design", r"branding", r"logos?"],
        "UI/UX Design": [r"ui/ux", r"ux", r"user experience", r"user interface", r"wireframes?", r"prototyp(?:e|es|ing)", r"figma", r"sketch"],
        "3D Modeling": [r"3d modeling", r"3ds max", r"maya", r"rhino", r"sketchup", r"revit", r"3d"],
        "Typography": [r"typography"],
        "Interior Design": [r"interior design", r"space planning"],
        "Fashion Design": [r"fashion", r"apparel", r"garments?", r"tech packs?"],
        "Web Design": [r"web design", r"wordpress", r"dreamweaver"],
    },
    "Healthcare": {
        "Patient Care": [r"patient care", r"patient", r"bedside"],
        "Medical Terminology": [r"medical terminology"],
        "EMR/EHR": [r"emr", r"ehr", r"electronic (?:medical|health) records?", r"epic", r"cerner", r"meditech"],
        "HIPAA": [r"hipaa"],
        "CPR/BLS": [r"cpr", r"bls", r"first aid", r"acls"],
        "Medical Billing": [r"medical billing", r"medical coding", r"icd-?(?:9|10)", r"cpt"],
        "Scheduling": [r"scheduling", r"appointments?"],
        "Pharmacy": [r"pharmacy", r"pharmaceutical", r"compounding"],
        "Nursing": [r"nursing", r"registered nurse", r"rn", r"lpn", r"cna"],
    },
    "Education": {
        "Teaching": [r"teach(?:ing|er)?", r"instruct(?:ion|or|ed)?", r"tutor(?:ing)?"],
        "Lesson Planning": [r"lesson plans?", r"lesson planning", r"curriculum"],
        "Classroom Management": [r"classroom management", r"classroom"],
        "Special Education": [r"special education", r"iep", r"special needs"],
        "Student Assessment": [r"assessments?", r"grading", r"standardized test(?:s|ing)?"],
        "Early Childhood Education": [r"early childhood", r"pre-?k", r"kindergarten", r"preschool"],
        "ESL": [r"esl", r"english as a second language", r"ell"],
        "Differentiated Instruction": [r"differentiated instruction", r"differentiat(?:e|ed|ion)"],
    },
}

# Flat views
SKILL_CATEGORY: dict[str, str] = {s: cat for cat, group in SKILLS.items() for s in group}
ALL_SKILLS: list[str] = list(SKILL_CATEGORY)


@lru_cache(maxsize=None)
def _pattern(skill: str) -> re.Pattern:
    aliases = SKILLS[SKILL_CATEGORY[skill]][skill]
    alt = "|".join(f"(?:{a})" for a in aliases)
    # custom boundaries so "c#", "c++" and ".net" still match
    return re.compile(rf"(?<!\w)(?:{alt})(?!\w)", re.IGNORECASE)


def find_skills(text: str) -> list[str]:
    """Return canonical skills mentioned in `text`, in taxonomy order."""
    return [s for s in ALL_SKILLS if _pattern(s).search(text)]


def skill_positions(text: str, skill: str) -> list[int]:
    return [m.start() for m in _pattern(skill).finditer(text)]


def normalize_skill(name: str) -> str | None:
    """Map a free-text skill name (e.g. 'MS Excel', 'postgres') to its canonical name."""
    name = name.strip()
    for s in ALL_SKILLS:
        if s.lower() == name.lower():
            return s
    hits = find_skills(name)
    return hits[0] if hits else None
