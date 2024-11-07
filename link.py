from typing import Dict, List, Optional, Tuple
import dedupe
import pandas as pd
import json
import logging
import csv
from csv import writer

from pandas.compat import os

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
    training_file = "training_file.json"

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

    linker = None
    if os.path.exists(settings_file):
        print(f"reading from settings file: {settings_file}")
        with open(settings_file, "rb") as sf:
            linker = dedupe.StaticRecordLink(sf)
    else:
        linker = dedupe.RecordLink(fields_for_matching)

        if os.path.exists(training_file):
            print("reading labeled examples from ", training_file)
            with open(training_file) as tf:
                linker.prepare_training(data_1, data_2, tf, sample_size=15000)
        else:
            linker.prepare_training(data_1, data_2, sample_size=5000)

        print("starting active labelling...")
        dedupe.console_label(linker)
        linker.train()
        with open(training_file, "w") as tf:
            linker.write_training(tf)

        with open(settings_file, "wb") as sf:
            linker.write_settings(sf)

    print("clustering...")
    linked_records = linker.join(data_1, data_2, threshold=0.8, constraint="one-to-one")

    print("# duplicate sets", len(linked_records))
    # ## Writing Results

    # Write our original data back out to a CSV with a new column called
    # 'Cluster ID' which indicates which records refer to each other.

    cluster_membership = {}
    for cluster_id, (cluster, score) in enumerate(linked_records):
        for record_id in cluster:
            cluster_membership[record_id] = {
                "cluster_id": cluster_id,
                "link_score": str(score),
            }
    left_file = places_data_file_name
    right_file = loans_data_file_name
    to_write_field_names = [
        "business_name",
        "zip",
        "address_clean",
        "city_clean",
        "state_clean",
    ]
    with open(output_file, mode="w", newline="", encoding="utf-8") as output_csv:
        fieldnames = to_write_field_names + ["cluster_id", "link_score", "source_file"]
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)

        writer.writeheader()
        matching_rows = {}
        for record_id, record_data in data_1.items():
            output_row = {
                field: record_data.get(field, "") for field in to_write_field_names
            }

            if record_id in cluster_membership:
                output_row.update(cluster_membership[record_id])
                output_row.update({"source_file": 0})
                matching_rows[cluster_membership[record_id]["cluster_id"]] = (
                    matching_rows.get(cluster_membership[record_id]["cluster_id"], []) + [output_row]
                )

            else:
                continue
            writer.writerow(output_row)

        for record_id, record_data in data_2.items():
            output_row = {
                field: record_data.get(field, "") for field in to_write_field_names
            }

            if record_id in cluster_membership:
                output_row.update(cluster_membership[record_id])
                output_row.update({"source_file": 1})

                matching_rows[cluster_membership[record_id]["cluster_id"]] = (
                    matching_rows.get(cluster_membership[record_id]["cluster_id"], []) + [output_row]
                )
            else:
                continue

            writer.writerow(output_row)

        with open("data/output/matching_rows.json", "w") as mr:
            mr.write(json.dumps(matching_rows, indent=2))
