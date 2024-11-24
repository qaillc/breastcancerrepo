import streamlit as st
import pandas as pd
import numpy as np

# Seed for reproducibility
np.random.seed(42)

# Function to generate synthetic data
def generate_realistic_data(num_patients=100):
    # Initialize data lists
    patient_ids = []
    ages = []
    menopausal_status = []
    tumor_sizes = []
    lymph_nodes = []
    grades = []
    stages = []
    er_status = []
    pr_status = []
    her2_status = []
    ki67_level = []
    tnbc_status = []
    brca_mutation = []
    overall_health = []
    genomic_score = []
    treatment = []

    for i in range(num_patients):
        # Patient ID
        patient_id = i + 1
        patient_ids.append(patient_id)

        # Age
        age = int(np.random.normal(60, 10))
        age = max(30, min(age, 80))
        ages.append(age)

        # Menopausal Status
        menopausal = 'Post-menopausal' if age >= 50 else 'Pre-menopausal'
        menopausal_status.append(menopausal)

        # Tumor Size
        tumor_size = round(np.random.lognormal(mean=0.7, sigma=0.5), 2)
        tumor_sizes.append(tumor_size)

        # Lymph Node Involvement
        lymph_node = 'Positive' if (tumor_size > 2.0 and np.random.rand() < 0.6) or (tumor_size <= 2.0 and np.random.rand() < 0.3) else 'Negative'
        lymph_nodes.append(lymph_node)

        # Tumor Grade
        grade = np.random.choice([1, 2, 3], p=[0.1, 0.4, 0.5] if tumor_size > 2.0 else [0.3, 0.5, 0.2])
        grades.append(grade)

        # Tumor Stage
        if tumor_size <= 2.0 and lymph_node == 'Negative':
            stage = 'I'
        elif (tumor_size > 2.0 and tumor_size <= 5.0) and lymph_node == 'Negative':
            stage = 'II'
        elif lymph_node == 'Positive' or tumor_size > 5.0:
            stage = 'III'
        else:
            stage = 'II'
        if np.random.rand() < 0.05:
            stage = 'IV'
        stages.append(stage)

        # Hormone Receptor Status
        er = np.random.choice(['Positive', 'Negative'], p=[0.75, 0.25])
        pr = 'Positive' if er == 'Positive' and np.random.rand() > 0.1 else 'Negative'
        er_status.append(er)
        pr_status.append(pr)

        # HER2 Status
        her2 = np.random.choice(['Positive', 'Negative'], p=[0.3, 0.7] if grade == 3 else [0.15, 0.85])
        her2_status.append(her2)

        # Ki-67 Level
        ki67 = 'High' if grade == 3 and np.random.rand() < 0.8 else 'Low'
        ki67_level.append(ki67)

        # Triple-Negative Status
        tnbc = 'Positive' if er == 'Negative' and pr == 'Negative' and her2 == 'Negative' else 'Negative'
        tnbc_status.append(tnbc)

        # BRCA Mutation
        brca = 'Positive' if (tnbc == 'Positive' or age < 40) and np.random.rand() < 0.2 else 'Negative'
        brca_mutation.append(brca)

        # Overall Health
        health = 'Good' if age < 65 and np.random.rand() < 0.9 else 'Poor'
        overall_health.append(health)

        # Genomic Recurrence Score
        recurrence_score = np.random.choice(['Low', 'Intermediate', 'High'], p=[0.6, 0.3, 0.1]) if er == 'Positive' and her2 == 'Negative' else 'N/A'
        genomic_score.append(recurrence_score)

        # Treatment
        if stage in ['I', 'II']:
            if tnbc == 'Positive':
                treat = 'Surgery, Chemotherapy, and Radiation Therapy' + (', plus PARP Inhibitors' if brca == 'Positive' else '')
            elif er == 'Positive' and recurrence_score != 'N/A':
                if recurrence_score == 'High':
                    treat = 'Surgery, Chemotherapy, Hormone Therapy, and Radiation Therapy'
                elif recurrence_score == 'Intermediate':
                    treat = 'Surgery, Consider Chemotherapy, Hormone Therapy, and Radiation Therapy'
                else:
                    treat = 'Surgery, Hormone Therapy, and Radiation Therapy'
            elif her2 == 'Positive':
                treat = 'Surgery, HER2-Targeted Therapy, Chemotherapy, and Radiation Therapy'
            else:
                treat = 'Surgery, Chemotherapy, and Radiation Therapy'
        elif stage == 'III':
            treat = 'Neoadjuvant Chemotherapy, Surgery, Radiation Therapy' + (', HER2-Targeted Therapy' if her2 == 'Positive' else '') + (', Hormone Therapy' if er == 'Positive' else '')
        else:
            treat = 'Systemic Therapy (' + ', '.join([option for option in ['Hormone Therapy' if er == 'Positive' else '', 'HER2-Targeted Therapy' if her2 == 'Positive' else '', 'Chemotherapy' if tnbc == 'Positive' else ''] if option]) + '), Palliative Care' if health == 'Good' else 'Palliative Care Only'

        treatment.append(treat)

    # Create DataFrame
    data = {
        'Patient ID': patient_ids,
        'Age': ages,
        'Menopausal Status': menopausal_status,
        'Tumor Size (cm)': tumor_sizes,
        'Lymph Node Involvement': lymph_nodes,
        'Tumor Grade': grades,
        'Tumor Stage': stages,
        'ER Status': er_status,
        'PR Status': pr_status,
        'HER2 Status': her2_status,
        'Ki-67 Level': ki67_level,
        'TNBC Status': tnbc_status,
        'BRCA Mutation': brca_mutation,
        'Overall Health': overall_health,
        'Genomic Recurrence Score': genomic_score,
        'Treatment': treatment,
    }

    return pd.DataFrame(data)

# Function to generate fuzzy data
def generate_fuzzy_data(df, error_rate=0.1):
    fuzzy_df = df.copy()
    num_rows, num_cols = fuzzy_df.shape
    
    # Introduce errors
    for _ in range(int(num_rows * num_cols * error_rate)):
        row = np.random.randint(0, num_rows)
        col = np.random.randint(0, num_cols)
        
        value = fuzzy_df.iloc[row, col]
        
        if isinstance(value, str):
            if value in ['Post-menopausal', 'Pre-menopausal']:
                fuzzy_df.iloc[row, col] = 'Post-menopausal' if value == 'Pre-menopausal' else 'Pre-menopausal'
            elif value in ['Positive', 'Negative']:
                fuzzy_df.iloc[row, col] = 'Negative' if value == 'Positive' else 'Positive'
        elif isinstance(value, (int, float)):
            noise = np.random.normal(0, 0.1 * value)
            fuzzy_df.iloc[row, col] += noise
    
    return fuzzy_df

def main():
    st.title('Synthetic Data Generator: Perfect and Fuzzy')
    st.write('This app generates synthetic breast cancer patient data and provides downloads for both perfect and fuzzy datasets.')
    
    num_patients = st.number_input('Number of Patients to Generate', min_value=10, max_value=10000, value=100, step=10)
    
    if st.button('Generate Data'):
        perfect_data = generate_realistic_data(num_patients)
        fuzzy_data = generate_fuzzy_data(perfect_data, error_rate=0.1)

        st.subheader('Perfect Data')
        st.dataframe(perfect_data)
        st.download_button('Download Perfect Data', perfect_data.to_csv(index=False), file_name='perfect_data.csv')

        st.subheader('Fuzzy Data (10% Error Rate)')
        st.dataframe(fuzzy_data)
        st.download_button('Download Fuzzy Data', fuzzy_data.to_csv(index=False), file_name='fuzzy_data.csv')

if __name__ == '__main__':
    main()
