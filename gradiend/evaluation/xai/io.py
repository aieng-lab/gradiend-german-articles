import itertools
import os.path
from enum import Enum

from tqdm import tqdm

from gradiend.evaluation.encoder.de_encoder_analysis import DeEncoderAnalysis
from gradiend.model import GradiendModel, ModelWithGradiend, gradiend_dir
from gradiend.util import RESULTS_DIR


class Gender(Enum):
    MALE = 1
    FEMALE = 2
    NEUTRAL = 3

class Case(Enum):
    NOMINATIVE = 1
    GENITIVE = 2
    ACCUSATIVE = 3
    DATIVE = 4


ALL_GENDERS = list(Gender)
ALL_CASES = list(Case)


enum2id = {
    Gender.FEMALE: 'F',
    Gender.MALE: 'M',
    Gender.NEUTRAL: 'N',
    Case.NOMINATIVE: 'N',
    Case.GENITIVE: 'G',
    Case.DATIVE: 'D',
    Case.ACCUSATIVE: 'A',
}

case_gender_mapping = {
    'N': {'F': 'die', 'M': 'der', 'N': 'das', 'P': 'die'}, # Nominativ
    'G': {'F': 'der', 'M': 'des', 'N': 'des', 'P': 'der'}, # Genitiv
    'A': {'F': 'die', 'M': 'den', 'N': 'das', 'P': 'die'}, # Akkusativ
    'D': {'F': 'der', 'M': 'dem', 'N': 'dem', 'P': 'den'}, # Dativ
}

article_mapping = {
    'NM': 'der',
    'NF': 'die',
    'NN': 'das',
    'AM': 'den',
    'AF': 'die',
    'AN': 'das',
    'DM': 'dem',
    'DF': 'der',
    'DN': 'dem',
    'GM': 'des',
    'GF': 'der',
    'GN': 'des',
}


pretty_dataset_mapping = {
    'NM': r'\dataNM',
    'NF': r'\dataNF',
    'NN': r'\dataNN',
    'AM': r'\dataAM',
    'AF': r'\dataAF',
    'AN': r'\dataAN',
    'DM': r'\dataDM',
    'DF': r'\dataDF',
    'DN': r'\dataDN',
    'GM': r'\dataGM',
    'GF': r'\dataGF',
    'GN': r'\dataGN',
}

latex_article_mapping = {
    pretty_dataset_mapping[ds]: article_mapping[ds] for ds in article_mapping.keys()
}
pretty_model_mapping = {
    'bert-base-german-cased': r'\bert',
    'gbert-large': r'\gbert',
    'german-gpt2': r'\gpttwo',
    'EuroBERT-210m': r'\eurobert',
    'Llama-3.2-3B': r'\llama',
    #'ModernGBERT_134M': r'\modernbert',
    'ModernGBERT_1B': r'\modernbert',
}

all_datasets = [
    'NM', 'NF', 'NN',
    'AM', 'AF', 'AN',
    'DM', 'DF', 'DN',
    'GM', 'GF', 'GN',
]


# each Configuration must have at least two genders OR two cases
class GradiendGenderCaseConfiguration:
    #def __init__(self, *config, mode='weight', part='decoder', model_id='german-gpt2', base_dir='results/experiments/gradiend', check_gradiend=False):
    def __init__(self, *config, mode='weight', part='decoder', model_id='bert-base-german-cased', base_dir=f'{RESULTS_DIR}/experiments/gradiend', check_gradiend=False):
        # Separate by type
        self.genders = list(sorted([c for c in config if isinstance(c, Gender)], key=lambda g: g.value)) or ALL_GENDERS
        self.cases = list(sorted([c for c in config if isinstance(c, Case)], key=lambda c: c.value)) or ALL_CASES

        # Check constraints
        if len(self.genders) < 2 and len(self.cases) < 2:
            raise ValueError("Configuration must contain at least two genders or two cases.")

        self.mode = mode
        self.part = part
        self.model_id = model_id
        self.base_dir = base_dir

        if check_gradiend and not os.path.isdir(self.gradiend_dir):
            raise FileNotFoundError(f'No Gradiend found at dir', self.gradiend_dir)

    @property
    def articles(self):
        articles = set()
        for case1, case2 in itertools.combinations_with_replacement(self.cases, 2):
            case1_str = enum2id[case1]
            case2_str = enum2id[case2]
            for gender1, gender2 in itertools.combinations_with_replacement(self.genders, 2):
                gender1_str = enum2id[gender1]
                gender2_str = enum2id[gender2]
                article1 = case_gender_mapping[case1_str][gender1_str]
                article2 = case_gender_mapping[case2_str][gender2_str]
                articles.add(article1)
                articles.add(article2)
        return list(sorted(articles))


    @property
    def is_gender_transition(self) -> bool:
        """
        True iff the configuration represents a gender transition
        (multiple genders, single case).
        """
        return len(self.genders) > 1 and len(self.cases) == 1

    @property
    def is_case_transition(self) -> bool:
        """
        True iff the configuration represents a case transition
        (multiple cases, single gender).
        """
        return len(self.cases) > 1 and len(self.genders) == 1

    @property
    def transition_type(self) -> str:
        if self.is_gender_transition:
            return "gender"
        if self.is_case_transition:
            return "case"
        raise ValueError(
            "Configuration must represent exactly one transition "
            "(either multiple genders or multiple cases)."
        )

    @property
    def local_rule_based_gender_case_article_pairs(self):
        return [(ds, art) for ds in self.datasets for art in self.articles]

    @property
    def memorization_gender_case_article_pairs(self):
        # returns all cells involved in the same article transition
        selected_articles = set(self.articles)

        ds_labels = article_mapping.keys()
        art_labels = list(sorted(set(article_mapping.values())))

        return {article: [
                (ds, art)
                for ds in ds_labels
                if article_mapping[ds] == article
                for art in art_labels
                if art in selected_articles
            ] for article in self.articles
        }

    @property
    def abstract_rule_based_gender_case_article_pairs(self):
        """
        Returns a list of (ds, article, direction) tuples.

        direction ∈ {"increase", "decrease"}
        """

        result = {}


        for article in self.articles:
            triples = set()

            # --------------------------------------------------
            # Gender transition -> propagate across ALL cases
            # --------------------------------------------------
            if self.is_gender_transition:
                # exactly two genders define the transition
                if len(self.genders) != 2:
                    raise ValueError("Gender transition must involve exactly two genders.")

                g_src, g_tgt = self.genders
                if article_mapping[enum2id[self.cases[0]] + enum2id[g_src]] != article:
                    g_src, g_tgt = g_tgt, g_src  # swap to ensure correct direction

                g_src_id = enum2id[g_src]
                g_tgt_id = enum2id[g_tgt]

                for case in ALL_CASES:
                    case_id = enum2id[case]

                    art_src = case_gender_mapping[case_id][g_src_id]
                    art_tgt = case_gender_mapping[case_id][g_tgt_id]

                    # source dataset
                    ds_src = case_id + g_src_id
                    triples.add((ds_src, art_src, "decrease"))
                    triples.add((ds_src, art_tgt, "increase"))

                    # target dataset
                    ds_tgt = case_id + g_tgt_id
                    triples.add((ds_tgt, art_src, "decrease"))
                    triples.add((ds_tgt, art_tgt, "increase"))

            # --------------------------------------------------
            # Case transition -> propagate across ALL genders
            # --------------------------------------------------
            elif self.is_case_transition:
                # exactly two cases define the transition
                if len(self.cases) != 2:
                    raise ValueError("Case transition must involve exactly two cases.")

                c_src, c_tgt = self.cases
                if article_mapping[enum2id[c_src] + enum2id[self.genders[0]]] != article:
                    c_src, c_tgt = c_tgt, c_src
                c_src_id = enum2id[c_src]
                c_tgt_id = enum2id[c_tgt]

                for gender in ALL_GENDERS:
                    gender_id = enum2id[gender]

                    art_src = case_gender_mapping[c_src_id][gender_id]
                    art_tgt = case_gender_mapping[c_tgt_id][gender_id]

                    # source dataset
                    ds_src = c_src_id + gender_id
                    triples.add((ds_src, art_src, "decrease"))
                    triples.add((ds_src, art_tgt, "increase"))

                    # target dataset
                    ds_tgt = c_tgt_id + gender_id
                    triples.add((ds_tgt, art_src, "decrease"))
                    triples.add((ds_tgt, art_tgt, "increase"))

            else:
                raise ValueError(
                    "Configuration must represent exactly one transition "
                    "(either gender or case)."
                )

            result[article] = list(sorted(triples))
        return result

    # todo
    @property
    def categories(self):
        return {}

    def get_model_metrics(self, split='test'):
        gradiend_path = self.gradiend_dir
        model_analyser = DeEncoderAnalysis({
            'categories': self.categories,
            'articles': self.articles,
            'plot_name': self.id,
            **{ds: {'encoding': 1 if ds == self.datasets[0] else (-1 if ds == self.datasets[1] else 0)} for ds in all_datasets}
        })
        model_metrics = model_analyser.get_model_metrics(gradiend_path, split=split)
        return model_metrics

    @property
    def article_changes(self):

        #if self.genders == ALL_GENDERS:

        #elif self.cases == ALL_CASES:

        article_changes = []
        for case1, case2 in itertools.combinations_with_replacement(self.cases, 2):
            case1_str = enum2id[case1]
            case2_str = enum2id[case2]
            for gender1, gender2 in itertools.combinations_with_replacement(self.genders, 2):
                gender1_str = enum2id[gender1]
                gender2_str = enum2id[gender2]
                article1 = case_gender_mapping[case1_str][gender1_str]
                article2 = case_gender_mapping[case2_str][gender2_str]
                if article1 != article2:
                    change = '<->'.join(sorted([article1, article2]))
                    article_changes.append(change)

        return list(sorted(set(article_changes)))

    @property
    def pretty_model_id(self):
        return rf'\gradc{"".join([enum2id[c] for c in self.cases])}g{"".join([enum2id[g] for g in self.genders])}'

    @property
    def articles_str(self):
        article_changes = self.article_changes
        output_str = ', '.join(sorted(set(article_changes)))
        if len(output_str) > 20:
            return output_str[:20] + '...'
        return output_str

    def __repr__(self):
        return f"GradiendGenderCaseConfiguration(genders={self.genders}, cases={self.cases})"

    @property
    def gradiend_dir(self):
        base_dir = f'{self.base_dir}/{self.id}/dim_1_inv_gradient/{self.model_id}'
        return gradiend_dir(base_dir)


    def load_gradiend(self, **kwargs):
        return GradiendModel.from_pretrained(self.gradiend_dir, **kwargs)

    def load_model_with_gradiend(self, **kwargs):
        return ModelWithGradiend.from_pretrained(self.gradiend_dir, **kwargs)

    @property
    def datasets(self):
        return [enum2id[case] + enum2id[gender] for case in self.cases for gender in self.genders]

    @property
    def id(self):
        gender_str = ""
        if self.genders != ALL_GENDERS:
            gender_str = "".join([enum2id[g] for g in self.genders])

        case_str = ""
        if self.cases != ALL_CASES:
            case_str = "".join([enum2id[c] for c in self.cases])

        intermediate_str = '_' if (gender_str and case_str) else ''

        return f'{case_str}{intermediate_str}{gender_str}_neutral_augmented'
        #return f'{case_str}{intermediate_str}{gender_str}'

def load_gradiends(*configs, return_pretty_model_id=False, **kwargs):
    gradiends = {}
    names = {}
    pretty_model_ids = {}
    for config in tqdm(configs, f"Loading GRADIENDs"):
        gradiend = config.load_gradiend(**kwargs)
        gradiends[config.id] = gradiend
        names[config.id] = f'{config.id}: {config.articles}'
        pretty_model_ids[config.id] = config.pretty_model_id

    output = gradiends, names

    if return_pretty_model_id:
        output = tuple(list(output) + [pretty_model_ids])

    return output

def load_model_with_gradiends(*configs, return_pretty_model_id=False, **kwargs):
    models_with_gradiends = {}
    names = {}
    pretty_model_ids = {}
    for config in tqdm(configs, f"Loading Models with GRADIENDs"):
        gradiend = config.load_model_with_gradiend(**kwargs)
        models_with_gradiends[config.id] = gradiend
        names[config.id] = f'{config.id}: {"-".join(sorted(config.articles))}'
        pretty_model_ids[config.id] = config.pretty_model_id

    output = models_with_gradiends, names
    if return_pretty_model_id:
        output = tuple(list(output) + [pretty_model_ids])

    return output



def load_all_gradiends():
    configs = []
    for case1, case2 in itertools.combinations(ALL_CASES, 2):
        configs.append(GradiendGenderCaseConfiguration(case1, case2))
        for gender in ALL_GENDERS:
            try:
                configs.append(GradiendGenderCaseConfiguration(case1, case2, gender, check_gradiend=True))
            except FileNotFoundError:
                pass  # some configurations are not defined

    for gender1, gender2 in itertools.combinations(ALL_GENDERS, 2):
        configs.append(GradiendGenderCaseConfiguration(gender1, gender2))
        for case in ALL_CASES:
            try:
                configs.append(GradiendGenderCaseConfiguration(gender1, gender2, case, check_gradiend=True))
            except FileNotFoundError:
                pass  # some configurations are not defined

    gradiends, names = load_gradiends(*configs)

    #gradiends['M:MF+MN'] = gradiends['MF'].normalize() + gradiends['MN'].normalize()
    #names['M:MF+MN'] = 'M:MF+MN'

    #gradiends['F:FN-MF'] = gradiends['FN'].normalize() - gradiends['MF'].normalize()
    #names['F:FN-MF'] = 'F:FN-MF'

    #gradiends['N:MN+FN'] = gradiends['MN'].normalize() + gradiends['FN'].normalize()
    #names['N:MN+FN'] = 'N:MN+FN'

    return gradiends, names


interesting_config_classes = {
    ('der', 'die'): [
        (Case.NOMINATIVE, Gender.MALE, Gender.FEMALE),
        (Case.NOMINATIVE, Case.DATIVE, Gender.FEMALE),
        (Case.NOMINATIVE, Case.GENITIVE, Gender.FEMALE),
        (Case.DATIVE, Case.ACCUSATIVE, Gender.FEMALE),
        (Case.GENITIVE, Case.ACCUSATIVE, Gender.FEMALE),
    ],
    ('der', 'des'): [
        (Case.NOMINATIVE, Case.GENITIVE, Gender.MALE),
        (Case.GENITIVE, Gender.FEMALE, Gender.MALE),
        (Case.GENITIVE, Gender.NEUTRAL, Gender.FEMALE),
    ],
    ('der', 'dem'): [
        (Case.NOMINATIVE, Case.DATIVE, Gender.MALE),
        (Case.DATIVE, Gender.MALE, Gender.FEMALE),
        (Case.DATIVE, Gender.NEUTRAL, Gender.FEMALE),
    ],
}

semi_interesting_classes = {
    ('die', 'das'): [
        (Case.NOMINATIVE, Gender.FEMALE, Gender.NEUTRAL),
        (Case.ACCUSATIVE, Gender.FEMALE, Gender.NEUTRAL),
    ],
    ('das', 'dem'): [
        (Case.NOMINATIVE, Case.DATIVE, Gender.NEUTRAL),
        (Case.ACCUSATIVE, Case.DATIVE, Gender.NEUTRAL),
    ],
    ('das', 'des'): [
        (Case.NOMINATIVE, Case.GENITIVE, Gender.NEUTRAL),
        (Case.ACCUSATIVE, Case.GENITIVE, Gender.NEUTRAL),
    ]
}

all_interesting_classes = interesting_config_classes | semi_interesting_classes

control_config = {
    'der_die_das_den': [
        (Case.ACCUSATIVE, Case.DATIVE, Gender.NEUTRAL),
        (Case.ACCUSATIVE, Case.DATIVE, Gender.FEMALE),
        (Case.ACCUSATIVE, Gender.NEUTRAL, Gender.FEMALE),
        (Case.DATIVE, Gender.NEUTRAL, Gender.FEMALE),
    ]
}


statistical_analysis_config_classes = {
    ('der', 'die'): [
        (Case.NOMINATIVE, Gender.MALE, Gender.FEMALE),
        #(Case.NOMINATIVE, Case.DATIVE, Gender.FEMALE),
        #(Case.NOMINATIVE, Case.GENITIVE, Gender.FEMALE),
        (Case.DATIVE, Case.ACCUSATIVE, Gender.FEMALE),
        (Case.GENITIVE, Case.ACCUSATIVE, Gender.FEMALE),
    ],
    ('der', 'dem'): [
        (Case.NOMINATIVE, Case.DATIVE, Gender.MALE),
        #(Case.DATIVE, Gender.MALE, Gender.FEMALE),
        (Case.DATIVE, Gender.NEUTRAL, Gender.FEMALE),
    ],
    ('der', 'des'): [
        (Case.NOMINATIVE, Case.GENITIVE, Gender.MALE),
        #(Case.GENITIVE, Gender.FEMALE, Gender.MALE),
        (Case.GENITIVE, Gender.NEUTRAL, Gender.FEMALE),
    ]
}


def load_gradiends_by_configs(configs, model_id='bert-base-german-cased', **kwargs):
    return load_gradiends(*[GradiendGenderCaseConfiguration(*config, model_id=model_id) for config in configs], **kwargs)

def load_model_with_gradiends_by_configs(configs, model_id='bert-base-german-cased', **kwargs):
    return load_model_with_gradiends(*[GradiendGenderCaseConfiguration(*config, model_id=model_id) for config in configs], **kwargs)

def load_der_die_gradiends():
    return load_gradiends_by_configs(interesting_config_classes[('der', 'die')])

def load_der_dem_gradiends():
    return load_gradiends_by_configs(interesting_config_classes[('der', 'dem')])

def load_der_des_gradiends():
    return load_gradiends_by_configs(interesting_config_classes[('der', 'des')])

def load_die_das_gradiends():
    return load_gradiends_by_configs(semi_interesting_classes[('die', 'das')])

def load_das_dem_gradiends():
    return load_gradiends_by_configs(semi_interesting_classes[('das', 'dem')])

def load_das_des_gradiends():
    return load_gradiends_by_configs(semi_interesting_classes[('das', 'des')])

def load_der_die_das_den_control_gradiends():
    configs = [
        GradiendGenderCaseConfiguration(Case.ACCUSATIVE, Case.DATIVE, Gender.NEUTRAL),
        GradiendGenderCaseConfiguration(Case.ACCUSATIVE, Case.DATIVE, Gender.FEMALE),
        GradiendGenderCaseConfiguration(Case.ACCUSATIVE, Gender.NEUTRAL, Gender.FEMALE),
        GradiendGenderCaseConfiguration(Case.DATIVE, Gender.NEUTRAL, Gender.FEMALE),

    ]# Dative



    return load_gradiends(*configs)

def load_all_gender_gradiends():
    configs = [
        GradiendGenderCaseConfiguration(Gender.MALE, Gender.FEMALE),
        GradiendGenderCaseConfiguration(Gender.MALE, Gender.NEUTRAL),
        GradiendGenderCaseConfiguration(Gender.FEMALE, Gender.NEUTRAL),

    ]

    return load_gradiends(*configs)


def load_all_case_gradiends():
    configs = [
        GradiendGenderCaseConfiguration(Case.NOMINATIVE, Case.GENITIVE),
        GradiendGenderCaseConfiguration(Case.NOMINATIVE, Case.DATIVE),
        GradiendGenderCaseConfiguration(Case.NOMINATIVE, Case.ACCUSATIVE),
        GradiendGenderCaseConfiguration(Case.GENITIVE, Case.DATIVE),
        GradiendGenderCaseConfiguration(Case.GENITIVE, Case.ACCUSATIVE),
        GradiendGenderCaseConfiguration(Case.DATIVE, Case.ACCUSATIVE),
    ]

    return load_gradiends(*configs)

def load_all_case_and_single_gender_gradiends():

    configs = []
    for case1, case2 in itertools.combinations(ALL_CASES, 2):
        configs.append(GradiendGenderCaseConfiguration(case1, case2))
        for gender in ALL_GENDERS:
            try:
                configs.append(GradiendGenderCaseConfiguration(case1, case2, gender, check_gradiend=True))
            except FileNotFoundError:
                pass # some configurations are not defined
    return load_gradiends(*configs)



if __name__ == '__main__':
    load_all_gender_gradiends()