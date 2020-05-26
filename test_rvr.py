import os
import time
import datetime

from skrvm import RVR

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pandas as pd
import numpy as np

from src import *


def normalize_df(df):
    f_min = min(df.min())
    f_max = max(df.max())

    return (df - f_min) / (f_max - f_min)


def train_and_test_model(
        feature_name,
        output_name,
        features, output, training_ids, test_ids,
        statistics,
):
    index_name = f"{feature_name}_to_{output_name}"
    printe(f"\n--- training {index_name} ---")

    train_features = features[features.index.map(lambda i: i in training_ids)]
    test_features = features[features.index.map(lambda i: i in test_ids)]

    train_output = output[output.index.map(lambda i: i in training_ids)][output_name]
    test_output = output[output.index.map(lambda i: i in test_ids)][output_name]

    if 1:
        start_time = time.time()

        #model = regression.RVRModel()
        #model = regression.LinearRegModel()
        model = regression.RidgeModel()

        model.fit(train_features.values, train_output.values)

        processing_time = round(time.time() - start_time, 1)
        print("  processing time", processing_time)

        sum_abs_error = []
        median_abs_error = []
        norm_abs_error = []
        for name, features, output in (
                ("training", train_features, train_output),
                ("test", test_features, test_output),
        ):
            prediction = model.predict(features)

            comparison = pd.DataFrame()
            comparison["index"] = output.index
            comparison["expected"] = output.values
            comparison["predicted"] = prediction
            comparison["error"] = output.values - prediction
            sum_abs_error.append(np.sum(np.abs(comparison["error"])))
            median_abs_error.append(sum_abs_error[-1] / comparison.shape[0])
            norm_abs_error.append(
                sum_abs_error[-1] / np.sum(output.values)
            )
            #print(test_output)
            #print(f"\n{name} evaluation on {feature_name}->{output_name}\n")
            # print(comparison)
            print(f"  {index_name}: {name} median abs error: {median_abs_error[-1]}, "
                  f"normalized abs error: {norm_abs_error[-1]}")

        statistics.loc[index_name] = {
            "dims": train_features.shape[1],
            "num_train": train_features.shape[0],
            "num_test": test_features.shape[0],
            "seconds": processing_time,
            "mae_train": median_abs_error[0],
            "mae_test": median_abs_error[1],
            "nae_train": norm_abs_error[0],
            "nae_test": norm_abs_error[1],
            "wnae_train": None,
            "wnae_test": None,
        }


def pca(features, num):
    printe(f"calc PCA {features.shape}")
    pca = PCA(n_components=num, svd_solver='full')
    values = pca.fit(features.values).transform(features.values)
    return pd.DataFrame(values, index=features.index)


def main():
    data = Data()
    statistics = pd.DataFrame(
        columns=(
            "dims", "num_train", "num_test", "seconds",
            "mae_train", "mae_test",
            "nae_train", "nae_test",
            "wnae_train", "wnae_test"
        ),
    )

    def _combine():
        return pca(
            pd.concat(
                (normalize_df(data.morphology), normalize_df(data.connectivity)),
                axis=1
            ),
            num=512
        )

    # for each modality
    for feature_name, full_features in (
            ("morphology", lambda : data.morphology),
            #("morphology_pca_26", lambda : pca(data.morphology, num=26)),
            #("connectivity_pca_1024", lambda : pca(data.connectivity, num=1024)),
            #("connectivity_pca_512", lambda : pca(data.connectivity, num=512)),
              #("connectivity_pca_128", lambda : pca(data.connectivity, num=128)),
            #("connectivity_pca_32", lambda : pca(data.connectivity, num=32)),
              #("morph_and_connectivity_pca_512", _combine),
            #("fmri_cut_slice_z", lambda : data.fmri_train("cut-slice-z")),
            #("fmri_act_series", lambda : data.fmri_train("act-series")),
              #("fmri_act_series_pca_128", lambda : pca(data.fmri_train("act-series"), 128)),
            #("fmri_1time_bucket_6th", lambda : data.fmri_train("1time-bucket-6th")),
            #("fmri_1time_bucket_6th_pca_32", lambda : pca(data.fmri_train("1time-bucket-6th"), 32)),
            #("fmri_1time_center_slice", lambda : data.fmri_train("1time-center-slice")),
            #("fmri_1time_center_slice_pca_32", lambda : pca(data.fmri_train("1time-center-slice"), 32)),
            #("fmri_1time_center_slice_pca_1024", lambda : pca(data.fmri_train("1time-center-slice"), 1024)),
            #("fmri_1time_desikan", lambda : data.fmri_train("1time-desikan")),
              #("fmri_1time_conv_bucket_6th", lambda : data.fmri_train("1time-conv-bucket-6th")),
              #("fmri_1time_conv_center_slice", lambda : data.fmri_train("1time-conv-center-slice")),
              #("fmri_1time_conv_desikan", lambda : data.fmri_train("1time-conv-desikan")),
            #("fmri_1time_diff_bucket_6th", lambda : data.fmri_train("1time-diff-bucket-6th")),
             #("fmri_1time_diff_bucket_6th_pca_512", lambda : pca(data.fmri_train("1time-diff-bucket-6th"))),
             #("fmri_1time_diff_center_slice", lambda : data.fmri_train("1time-diff-center-slice")),
            #("fmri_sum_bucket_6th", lambda : data.fmri_train("sum-bucket-6th")),
            #("fmri_sum_bucket_6th_pca_32", lambda : pca(data.fmri_train("sum-bucket-6th"), 32)),
              #("fmri_sum_desikan", lambda : data.fmri_train("sum-desikan")),
              #("fmri_sum_desikan_pca_64", lambda : pca(data.fmri_train("sum-desikan"), 64)),
            #("fmri_sum_center_slice", lambda : data.fmri_train("sum-center-slice")),
              #("fmri_sum_conv_bucket_6th", lambda : data.fmri_train("sum-conv-bucket-6th")),
              #("fmri_sum_conv_center_slice", lambda : data.fmri_train("sum-conv-center-slice")),
              #("fmri_sum_conv_desikan", lambda : data.fmri_train("sum-conv-desikan"))
            #("fmri_halftime_center_slice", lambda : data.fmri_train("halftime-center-slice")),
    ):
        full_features = full_features()

        # for each related output variable
        for output_name in data.training.keys():

            features = full_features[full_features.index.map(lambda i: i in data.training_ids)]

            # make sure there's no NaN in output data
            valid_rows = pd.notna(data.training[output_name])
            training = data.training[valid_rows]
            features = features[valid_rows]

            if 1:
                LIMIT = 10000
                training_ids = training.index[:LIMIT:2]
                test_ids = training.index[1:LIMIT:2]
            else:
                training_ids = training.index[:5000]
                test_ids = training.index[5000:]

            train_and_test_model(
                feature_name=feature_name,
                output_name=output_name,
                features=normalize_df(features),
                output=training,
                training_ids=training_ids,
                test_ids=test_ids,
                statistics=statistics,
            )

        # gather weighted normalized abs error for all modalities
        wnae = []
        for set_name in ("train", "test"):
            column = statistics[f"nae_{set_name}"]
            wnae.append(
                column[-1] * 0.175 + column[-2] * 0.175 + column[-3] * 0.175 + column[-4] * 0.175 \
                + column[-5] * 0.3
            )

        statistics.loc[feature_name] = {
            "dims": None,
            "num_train": None,
            "num_test": None,
            "seconds": None,
            "mae_train": None,
            "mae_test": None,
            "nae_train": None,
            "nae_test": None,
            "wnae_train": wnae[0],
            "wnae_test": wnae[1],
        }

        del full_features

    print("\n--- stats ---")
    print(repr(statistics))

    filename = "rvr-test-%s.csv" % datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    statistics.to_csv(
        os.path.join(STATS_DIR, filename)
    )
    printe(f"stored to {filename}")


if __name__ == "__main__":
    main()
    #print(pd.notna(Data().training["domain1_var1"]))
