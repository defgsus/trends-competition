import os
import time
import datetime

from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

from src import *


def normalize_df(df):
    f_min = min(df.min())
    f_max = max(df.max())

    return (df - f_min) / (f_max - f_min)


def pca(features, num=10):
    pca = PCA(n_components=num, svd_solver='full')
    values = pca.fit(features.values).transform(features.values)
    return pd.DataFrame(values, index=features.index)


def train_model(
        Model,
        feature_name, output_name,
        train_features, train_output,
):
    index_name = f"{feature_name}_to_{output_name}"
    printe(f"\n--- training {index_name} ({len(train_features.index)} subjects) ---")

    start_time = time.time()

    model = Model()
    model.fit(train_features.values, train_output[output_name].values)

    processing_time = round(time.time() - start_time, 1)
    printe("  processing time", processing_time)

    return model


def predict_with_rvr(rvr, test_features, submission_data, output_name):
    printe(f"\n--- predicting {len(test_features.index)} subjects ---")

    prediction = rvr.predict(test_features)

    for id, value in zip(test_features.index, prediction):
        submission_data[f"{id}_{output_name}"] = value


def main():
    data = Data()

    submission_data = dict()

    start_time = time.time()

    # for each modality
    for Model, output_name, feature_name, full_features in (
            (regression.RVRModel, "age",  "morphology_pca_26", lambda : pca(data.morphology, num=26)),
            (regression.LinearRegModel, "domain1_var1", "morphology", lambda : data.morphology),
            (regression.LinearRegModel, "domain1_var2", "connectivity_pca_512", lambda : pca(data.connectivity, num=512)),
            #(regression.RVRModel, "domain1_var2", "fmri_1time_center_slice", lambda : data.fmri("1time-center-slice")),
            (regression.RVRModel, "domain2_var1", "morphology", lambda : data.morphology),
            (regression.RVRModel, "domain2_var2", "connectivity_pca_32", lambda : pca(data.connectivity, num=32)),
             #("connectivity_pca_1024", lambda : pca(data.connectivity, num=1024)),
             #("connectivity_pca_512", lambda : pca(data.connectivity, num=512)),
             #("connectivity_pca_128", lambda : pca(data.connectivity, num=128)),
             #("connectivity_pca_32", lambda : pca(data.connectivity, num=32)),
             #("morph_and_connectivity_pca_512", _combine),
    ):
        full_features = full_features()

        LIMIT = 100000

        features = full_features[full_features.index.map(lambda i: i in data.training_ids[:LIMIT])]
        training = data.training[data.training.index.map(lambda i: i in data.training_ids[:LIMIT])]

        # make sure there's no NaN in output data
        valid_rows = pd.notna(data.training[output_name])
        features = features[valid_rows]
        training = training[valid_rows]

        model = train_model(
            Model=Model,
            feature_name=feature_name,
            output_name=output_name,
            train_features=normalize_df(features),
            train_output=training,
        )

        test_features = full_features[full_features.index.map(lambda i: i in data.test_ids[:LIMIT])]

        predict_with_rvr(model, normalize_df(test_features), submission_data, output_name)

    proc_time = time.time() - start_time

    # print(submission_data)

    submission_index = []
    submission_predicted = []
    for id in sorted(submission_data.keys()):
        submission_index.append(id)
        submission_predicted.append(submission_data[id])

    submission = pd.DataFrame(
        columns=(
            "Predicted",
        ),
        index=submission_index,
        data={"Predicted": submission_predicted}
    )
    submission = submission.reindex(submission.index.rename("Id"))
    print(submission)

    filename = f"submission-{feature_name}-%s.csv" % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    submission.to_csv(
        os.path.join(SUBMISSION_DIR, filename)
    )
    printe(f"stored to {filename}")

    printe(f"took {round(proc_time, 2)} seconds")


if __name__ == "__main__":
    main()
    #print(pd.notna(Data().training["domain1_var1"]))
