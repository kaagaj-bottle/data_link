from typing import Dict, List, Optional, Tuple
import dedupe
import pandas as pd
import json
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("dedupe_linking.log"), logging.StreamHandler()],
)


def read_json(filename):
    content = None

    with open(filename) as fp:
        content = json.loads(fp.read())

    return content


def get_field_from_dict(data: Optional[Dict], field_name_tuple):
    field_name: str = field_name_tuple
    if isinstance(data, dict):
        return data.get(field_name, None)
    return None


def middleware(func, field_name):
    return func(field_name)


def restructure_common_fields(
    df: pd.DataFrame, old_fields: List[str], new_fields: List[str], df_dict_key: str
):
    for idx in range(len(new_fields)):
        df[new_fields[idx]] = df[df_dict_key].apply(
            func=get_field_from_dict, args=[old_fields[idx]]
        )

    return df


def process_zip_for_loans_df_gov(zip):
    try:
        zip_str = str(int(zip))
        return zip_str
    except Exception as _:
        return None


def strip_str(zip: str):
    return zip.split(".")[0]


def combine_data(data_1, data_2):
    combined_data = {}
    for k, v in data_1.items():
        combined_data[f"a_{k}"] = v
    for k, v in data_2.items():
        combined_data[f"a_{k}"] = v

    return combined_data


if __name__ == "__main__":
    output_file = "data/output/data_matching_output.csv"
    settings_file = "data/output/data_matching_learned_settings"
    training_file = "data/output/data_matching_training.json"

    # reading data
    loans_data_file_name = "data/input/ppp_loans_state_CO.csv"
    places_data_file_name = "data/input/places.json"

    loans_df_gov = pd.read_csv(loans_data_file_name).astype(str)
    loans_df_gov["zip"] = loans_df_gov["zip"].astype(str).apply(strip_str)

    places_df_audit_city = pd.DataFrame(read_json(places_data_file_name)["data"])
    places_df_audit_city["business_name"] = places_df_audit_city["name"]

    old_fields = ["street", "city", "adm1", "postcode"]
    new_fields = ["address_clean", "city_clean", "state_clean", "zip"]
    df_dict_key = "address"

    places_df_audit_city = restructure_common_fields(
        places_df_audit_city, old_fields, new_fields, df_dict_key
    )

    fields_for_matching = [
        dedupe.variables.String("business_name"),
        dedupe.variables.Exact("zip"),
        dedupe.variables.String("address_clean"),
        dedupe.variables.String("city_clean"),
        dedupe.variables.String("state_clean"),
    ]

    # transforming pandas DataFrame to python dictionary
    data_1 = places_df_audit_city.to_dict(orient="index")
    data_2 = loans_df_gov.to_dict(orient="index")

    combined_data = combine_data(data_1, data_2)
    logging.info("Initializing Dedupe for record linkage.")
    linker = dedupe.RecordLink(fields_for_matching)

    logging.info("Starting active learning sampling.")
    linker.prepare_training(data_1, data_2, sample_size=5000)

    dedupe.console_label(linker)
    with open(settings_file,"wb") as sf:
        linker.write_settings(sf)


    linker.train()
    with open(training_file, "w") as tf:
        linker.write_training(tf)

