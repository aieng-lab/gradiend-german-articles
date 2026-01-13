from gradiend.evaluation.xai.io import Gender, Case, GradiendGenderCaseConfiguration, pretty_model_mapping
import numpy as np

configs = [
    # der ↔ die
    (Case.NOMINATIVE, Gender.MALE,   Gender.FEMALE),
    (Case.NOMINATIVE, Case.DATIVE, Gender.FEMALE),
    (Case.NOMINATIVE, Case.GENITIVE,   Gender.FEMALE),
    (Case.ACCUSATIVE, Case.DATIVE,   Gender.FEMALE),
    (Case.ACCUSATIVE,   Case.GENITIVE,   Gender.FEMALE),

    # der ↔ dem
    (Case.DATIVE,     Gender.MALE,   Gender.FEMALE),
    (Case.DATIVE, Gender.NEUTRAL,   Gender.FEMALE),
    (Case.NOMINATIVE, Case.DATIVE,   Gender.FEMALE),

    # der ↔ des
    (Case.GENITIVE,   Gender.FEMALE, Gender.MALE),
    (Case.GENITIVE,   Gender.FEMALE, Gender.NEUTRAL),
    (Case.NOMINATIVE, Case.GENITIVE,   Gender.MALE),

    # die ↔ das
    (Case.NOMINATIVE, Gender.FEMALE, Gender.NEUTRAL),
    (Case.ACCUSATIVE, Gender.FEMALE, Gender.NEUTRAL),

    # das ↔ dem
    (Case.NOMINATIVE, Case.DATIVE, Gender.NEUTRAL),
    (Case.ACCUSATIVE, Case.DATIVE, Gender.NEUTRAL),

    # das ↔ des
    (Case.NOMINATIVE, Case.GENITIVE, Gender.NEUTRAL),
    (Case.ACCUSATIVE, Case.GENITIVE, Gender.NEUTRAL),
]


models = [
    'bert-base-german-cased',
    'gbert-large',
    'ModernGBERT_1B',
    'EuroBERT-210m',
    'german-gpt2',
    'Llama-3.2-3B',
]

def de_encoder_table():

    table_data = []

    total_entries = 0
    filled_entries = 0
    for model in models:
        table_data_row = [pretty_model_mapping[model]]
        for config in configs:
            total_entries += 1
            gradiend_config = GradiendGenderCaseConfiguration(*config, model_id=model)
            try:
                print(f"Processing: {gradiend_config.id}")
                metrics = gradiend_config.get_model_metrics()
                p = metrics['pearson']
                table_data_row.append(f"{abs(p) * 100:.1f}")
                filled_entries += 1
            except Exception as e:
                print('Error: ', e)
                table_data_row.append("N/A")

        table_data.append("\t& ".join(table_data_row) + "\\\\")

    print('\n'.join(table_data))

    print(f"Filled entries: {filled_entries}/{total_entries} ({(filled_entries/total_entries)*100:.1f}%)")


if __name__ == '__main__':
    de_encoder_table()




