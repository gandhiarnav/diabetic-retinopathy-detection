def generate_clinical_summary(predicted_grade):
    """
    Maps the predicted DR grade (0-4) to standard clinical recommendations.
    """
    guidelines = {
        0: {
            "Diagnosis": "No Diabetic Retinopathy",
            "Timeline": "12 months",
            "Action": "No immediate action required. Maintain good glycemic control and schedule a routine annual eye exam."
        },
        1: {
            "Diagnosis": "Mild Non-Proliferative DR",
            "Timeline": "6 to 12 months",
            "Action": "Monitor closely. Ensure strict control of blood sugar, blood pressure, and cholesterol. Schedule a follow-up exam in 6-12 months."
        },
        2: {
            "Diagnosis": "Moderate Non-Proliferative DR",
            "Timeline": "3 to 6 months",
            "Action": "Disease progression noted. Referral to an ophthalmologist or retina specialist is recommended for a dilated exam within 3-6 months."
        },
        3: {
            "Diagnosis": "Severe Non-Proliferative DR",
            "Timeline": "1 to 3 months",
            "Action": "High risk of vision loss. Urgent referral to a retina specialist. Treatment (like laser therapy or injections) may be required soon."
        },
        4: {
            "Diagnosis": "Proliferative Diabetic Retinopathy",
            "Timeline": "Immediate / Urgent",
            "Action": "CRITICAL: Immediate referral to a retina specialist. Urgent intervention required to prevent irreversible blindness."
        }
    }
    
    return guidelines.get(predicted_grade, {"Diagnosis": "Unknown", "Timeline": "Consult Doctor", "Action": "Error reading grade."})
