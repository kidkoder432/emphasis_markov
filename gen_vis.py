from generate_fsm import *
import generate_fsm as gf
from training.learn_emphasis import *
from pprint import pprint

gf.changeRaga("amrit")

notes = raga_data["notes"]
gen = create_tags()

gf.ENABLE_EMPHASIS = True
gf.ENABLE_TAGS = True
gf.ENABLE_HYBRID = True

phrases, _ = generate_all_phrases(swars_list, convolved_splines, True, gen)

pprint(phrases)

for i, p in enumerate(phrases):
    phrases[i] = " ".join(p.split(" ")[3:])
phrase_data = {
    "notes": notes,
    "phrases": phrases,
}


emphasis_df = calculate_emphasis_df(phrase_data)

class NearestRowLookup:
    def __init__(self, df, value_col):
        """
        df: your DataFrame
        index_col: column to match against
        value_col: column to return
        """
        self.df = df
        self.index = df.index.to_numpy()
        self.values = df[value_col].values

    def __call__(self, x):
        x = np.atleast_1d(x)  # allow scalar or array
        idx = np.floor(x).astype(int)
        result = self.values[idx]
        if result.size == 1:
            return result[0]  # return scalar if input was scalar
        return result

splines = [NearestRowLookup(emphasis_df, c) for c in emphasis_df.columns]

t_gen, _, splines_gen = generate_splines(emphasis_df)

generate_plots_and_save_splines(emphasis_df, t_gen, None, splines_gen, "outputs/splines.pkl")
