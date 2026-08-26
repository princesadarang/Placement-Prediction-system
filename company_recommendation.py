def suggest_companies(
    cgpa=0,
    aptitude_score=0,
    technical_skill_score=0,
    communication_skill_score=0,
    coding_score=0,
    internship_experience=0,
    projects=0,
    backlogs=0,
    certifications=0
):

    
    # SAFE CONVERSION
    

    try:
        cgpa = float(cgpa or 0)

        aptitude_score = float(
            aptitude_score or 0
        )

        technical_skill_score = float(
            technical_skill_score or 0
        )

        communication_skill_score = float(
            communication_skill_score or 0
        )

        coding_score = float(
            coding_score or 0
        )

        internship_experience = int(
            internship_experience or 0
        )

        projects = int(
            projects or 0
        )

        backlogs = int(
            backlogs or 0
        )

        certifications = int(
            certifications or 0
        )

    except (ValueError, TypeError):

        return []


    companies = []


    
    # TOP PRODUCT COMPANIES
    
    if (
        cgpa >= 8
        and aptitude_score >= 75
        and technical_skill_score >= 80
        and coding_score >= 80
    ):

        companies.extend([
            "Google",
            "Microsoft",
            "Amazon",
            "Adobe",
            "Oracle"
        ])


   
    # GOOD IT COMPANIES
    
    if (
        cgpa >= 7
        and aptitude_score >= 60
        and technical_skill_score >= 65
        and coding_score >= 60
    ):

        companies.extend([
            "TCS",
            "Infosys",
            "Accenture",
            "Wipro",
            "Cognizant",
            "Capgemini"
        ])


   
    # ENTRY LEVEL
    
    if (
        cgpa >= 6
        and aptitude_score >= 50
        and technical_skill_score >= 50
        and backlogs <= 2
    ):

        companies.extend([
            "HCLTech",
            "Tech Mahindra",
            "LTIMindtree",
            "Mphasis",
            "Persistent Systems"
        ])


    
    # SOFTWARE DEVELOPMENT
    

    if (
        coding_score >= 70
        and projects >= 3
        and technical_skill_score >= 70
    ):

        companies.extend([
            "Zoho",
            "Freshworks"
        ])


    # =====================================================
    # AI / ML PROFILE
    # =====================================================

    if (
        technical_skill_score >= 75
        and projects >= 3
        and certifications >= 2
    ):

        companies.extend([
            "AI/ML Startups",
            "Data Science Startups",
            "AI Product Companies"
        ])


    
    # INTERNSHIP
    

    if internship_experience == 0:

        companies.append(
            "Internship Programs"
        )


    
    # REMOVE DUPLICATES
    

    companies = list(
        dict.fromkeys(companies)
    )


    return companies