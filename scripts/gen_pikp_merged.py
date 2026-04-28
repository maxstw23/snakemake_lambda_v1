"""
Generate pikp_merged.py from Etabins_20_Coalescence txt datapoint files.
Run from the project root:
    python scripts/gen_pikp_merged.py
"""

import re
import pathlib

TXT_DIR = pathlib.Path("Etabins_20_Coalescence/Etabins_20_Coalescence_ymp6top6/txtfiles")

# Map txt-file energy prefix → pikp_merged.py key
ENERGY_MAP = {
    "7GeV":  "7.7GeV",
    "9GeV":  "9.2GeV",
    "11GeV": "11.5GeV",
    "14GeV": "14.6GeV",
    "17GeV": "17.3GeV",
    "19GeV": "19.6GeV",
    "27GeV": "27GeV",
}

PARTICLES = ["pions", "kaons", "protons"]

# Combined-centrality order in the txt files: [0-10, 10-40, 40-80, 50-80]
COMBINED_CENT_KEYS = ["010", "1040", "4080", "5080"]


def parse_array(content, varname):
    """Extract a C-style double array from file content."""
    pattern = rf"double {re.escape(varname)}\[\d+\]=\{{([^}}]+)\}}"
    m = re.search(pattern, content)
    if not m:
        raise ValueError(f"Variable not found: {varname}")
    return [float(x) for x in m.group(1).split(",")]


def parse_txt(path):
    """Parse one datapoints txt file and return a dict with all needed arrays."""
    content = path.read_text()
    d = {}
    for fit in ("linear", "cubic"):
        d[f"pos_{fit}"]          = parse_array(content, f"v1_vCent_Selp_{fit}")
        d[f"pos_{fit}_err"]      = parse_array(content, f"v1_vCent_Selp_{fit}_err")
        d[f"pos_{fit}_systematics"] = parse_array(content, f"v1_vCent_Selp_{fit}_systematics")
        d[f"neg_{fit}"]          = parse_array(content, f"v1_vCent_Seln_{fit}")
        d[f"neg_{fit}_err"]      = parse_array(content, f"v1_vCent_Seln_{fit}_err")
        d[f"neg_{fit}_systematics"] = parse_array(content, f"v1_vCent_Seln_{fit}_systematics")
        d[f"delta_{fit}"]        = parse_array(content, f"deltav1_vCent_{fit}")
        d[f"delta_{fit}_err"]    = parse_array(content, f"deltav1_vCent_{fit}_err")
        d[f"delta_{fit}_systematics"] = parse_array(content, f"deltav1_vCent_{fit}_systematics")

        pos_comb  = parse_array(content, f"v1slopes_{fit}_combinedcent_pos")
        pos_comb_err  = parse_array(content, f"v1slopes_{fit}_combinedcent_pos_err")
        pos_comb_sys  = parse_array(content, f"v1slopes_{fit}_combinedcent_pos_systematics")
        neg_comb  = parse_array(content, f"v1slopes_{fit}_combinedcent_neg")
        neg_comb_err  = parse_array(content, f"v1slopes_{fit}_combinedcent_neg_err")
        neg_comb_sys  = parse_array(content, f"v1slopes_{fit}_combinedcent_neg_systematics")
        dv1_comb  = parse_array(content, f"v1slopes_{fit}_combinedcent_deltav1")
        dv1_comb_err  = parse_array(content, f"v1slopes_{fit}_combinedcent_deltav1_err")
        dv1_comb_sys  = parse_array(content, f"v1slopes_{fit}_combinedcent_deltav1_systematics")

        for i, cent in enumerate(COMBINED_CENT_KEYS):
            d[f"{cent}_{fit}"] = {
                "pos":              pos_comb[i],
                "pos_err":          pos_comb_err[i],
                "pos_systematics":  pos_comb_sys[i],
                "neg":              neg_comb[i],
                "neg_err":          neg_comb_err[i],
                "neg_systematics":  neg_comb_sys[i],
                "delta":            dv1_comb[i],
                "delta_err":        dv1_comb_err[i],
                "delta_systematics": dv1_comb_sys[i],
            }
    return d


def fmt_array(values):
    return "np.array([" + ", ".join(repr(v) for v in values) + "])"


def fmt_dict(d):
    items = ", ".join(f"'{k}': {repr(v)}" for k, v in d.items())
    return "{" + items + "}"


def main():
    # Collect all data
    data = {}
    for txt_energy, py_energy in ENERGY_MAP.items():
        data[py_energy] = {}
        for particle in PARTICLES:
            path = TXT_DIR / f"{txt_energy}_{particle}_datapoints.txt"
            data[py_energy][particle] = parse_txt(path)

    # Render pikp_merged.py
    lines = [
        "import numpy as np",
        "",
        "class PikpMergedSlope:",
        "    def __init__(self):",
        "        self.data = {",
    ]

    for py_energy, particles in data.items():
        lines.append(f"\t\t\t'{py_energy}': {{")
        for particle, d in particles.items():
            lines.append(f"\t\t\t\t'{particle}': {{")
            for fit in ("linear", "cubic"):
                for sign in ("pos", "neg", "delta"):
                    key = f"{sign}_{fit}"
                    lines.append(f"\t\t\t\t\t'{key}': {fmt_array(d[key])},")
                    lines.append(f"\t\t\t\t\t'{key}_err': {fmt_array(d[key+'_err'])},")
                    lines.append(f"\t\t\t\t\t'{key}_systematics': {fmt_array(d[key+'_systematics'])},")
                for cent in COMBINED_CENT_KEYS:
                    ckey = f"{cent}_{fit}"
                    lines.append(f"\t\t\t\t\t'{ckey}': {fmt_dict(d[ckey])},")
            lines.append("\t\t\t\t},")
        lines.append("\t\t\t},")

    lines += [
        "        }",
        "",
        "",
        "    def get_data(self):",
        "        return self.data",
    ]

    out = pathlib.Path("scripts/pikp_merged.py")
    out.write_text("\n".join(lines) + "\n")
    print(f"Written: {out}")


if __name__ == "__main__":
    main()
