import hashlib
import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import polars as pl
import pymc as pm
from jp_imports import JPTrade
from jp_tools import download
from thefuzz import fuzz, process


class ProdRepl(JPTrade):
    def __init__(self, saving_dir: str = "data/", database_file: str = "data.ddb"):
        super().__init__(saving_dir, database_file)

    def data_qcew(self):
        df_qcew = self.conn.execute(f"""
                    SELECT
                        year,
                        qtr,
                        phys_addr_5_zip,
                        phys_addr_city,
                        ui_addr_5_zip,
                        mail_addr_5_zip,
                        ein,
                        first_month_employment,
                        total_wages,
                        second_month_employment,
                        third_month_employment,
                        naics_code
                    FROM '{self.saving_dir}processed/pr-qcew-*.parquet' 
                    WHERE year != 2002;
                    """).pl()

        df_qcew = df_qcew.with_columns(
            first_month_employment=pl.col("first_month_employment").fill_null(
                strategy="zero"
            ),
            second_month_employment=pl.col("second_month_employment").fill_null(
                strategy="zero"
            ),
            third_month_employment=pl.col("third_month_employment").fill_null(
                strategy="zero"
            ),
            total_wages=pl.col("total_wages").fill_null(strategy="zero"),
            sector_code=pl.col("naics_code").str.slice(0, 2),
        )
        df_qcew = df_qcew.with_columns(
            total_employment=(
                pl.col("first_month_employment")
                + pl.col("second_month_employment")
                + pl.col("third_month_employment")
            )
            / 3,
        )

        df_qcew = df_qcew.filter(
            (pl.col("total_employment") != 0) & (pl.col("total_wages") != 0)
        )

        df_qcew = df_qcew.filter(
            (pl.col("phys_addr_city") != "")
            & (pl.col("naics_code") != "")
            & ((pl.col("year") < 2024))
        )
        df_muni_monthly = (
            df_qcew.select(
                [
                    "year",
                    "qtr",
                    "sector_code",
                    "phys_addr_city",
                    "first_month_employment",
                    "second_month_employment",
                    "third_month_employment",
                ]
            )
            # Pivot from columns to rows to get a true 'month' column
            .unpivot(
                index=["year", "qtr", "sector_code", "phys_addr_city"],
                on=[
                    "first_month_employment",
                    "second_month_employment",
                    "third_month_employment",
                ],
                variable_name="month_of_qtr",
                value_name="employment",
            ).with_columns(
                month=pl.when(pl.col("month_of_qtr") == "first_month_employment")
                .then((pl.col("qtr") - 1) * 3 + 1)
                .when(pl.col("month_of_qtr") == "second_month_employment")
                .then((pl.col("qtr") - 1) * 3 + 2)
                .otherwise((pl.col("qtr") - 1) * 3 + 3)
            )
        )

        df_muni_monthly = df_muni_monthly.group_by(
            ["year", "month", "sector_code", "qtr", "phys_addr_city"]
        ).agg(pl.col("employment").sum())

        df_muni_monthly = df_muni_monthly.with_columns(
            name=(
                pl.col("phys_addr_city")
                .str.normalize("NFD")
                .str.replace_all(r"[\u0300-\u036f]", "")
                .str.to_lowercase()
                .str.replace_all("  ", "_")
                .str.replace_all(" ", "_")
            )
        )

        return df_muni_monthly

    def data_imports(self) -> pl.DataFrame:
        df_imports = self.process_int_jp(level="naics", time_frame="monthly")
        df_imports = df_imports.with_columns(
            sector_code=pl.col("naics").str.slice(0, 2)
        )
        df_imports = df_imports.group_by(["year", "month", "sector_code"]).agg(
            pl.col(
                "imports",
                "imports_qty",
                "exports",
                "exports_qty",
                "net_exports",
                "net_qty",
            ).sum()
        )
        return df_imports

    def county_geom(self) -> gpd.GeoDataFrame:
        """
        Retrieves and processes US County TIGER/Line shapefiles with region classifications.

        Downloads the 2025 Census county geometry zip file if not present locally,
        filters for Puerto Rico (FIPS '72'), joins with internal region mapping,
        and caches the result as a Parquet file for future use.

        Returns
        -------
        gpd.GeoDataFrame
            A GeoDataFrame containing county geometries with the following columns:
            'statefip', 'geoid', 'name', 'geometry', and 'region'.

        Notes
        -----
        The method uses a temporary zip file for the initial download to keep the
        local storage directory clean. The internal region mapping is loaded from
        the package resources.
        """
        file_path = Path(self.saving_dir) / "external" / "geo-county.parquet"

        # Create a unique hash for the temp file based on the target path
        name_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        temp_zip = Path(tempfile.gettempdir()) / f"{name_hash}.zip"

        if not file_path.exists():
            # Ensure the directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)

            download(
                url="https://www2.census.gov/geo/tiger/TIGER2025/COUNTY/tl_2025_us_county.zip",
                filename=str(temp_zip),
            )

            # Process shape
            gdf = gpd.read_file(temp_zip)
            gdf = gdf.rename(
                columns={
                    "STATEFP": "statefip",
                    "GEOID": "geoid",
                    "NAME": "name",
                }
            )
            # Filter for Puerto Rico (FIPS 72)
            gdf = gdf[gdf["statefip"] == "72"].reset_index()
            gdf = gdf[["statefip", "geoid", "name", "geometry"]]

            gdf.to_parquet(file_path)
        return gpd.read_parquet(path=file_path)

    def reg_data(self):
        file_path = Path(self.saving_dir) / "raw" / "data.parquet"

        if file_path.exists():
            df_qcew = self.data_qcew().to_pandas()
            gdf = self.county_geom()[["name", "geoid"]]
            gdf = pl.DataFrame(gdf)
            gdf = gdf.with_columns(
                pl.col("name")
                .str.normalize("NFD")
                .str.replace_all(r"[\u0300-\u036f]", "")
                .str.to_lowercase()
                .str.replace_all(" ", "_")
            )
            gdf = gdf.to_pandas()
            # 2. Get the list of 'correct' names from df2
            correct_names = gdf["name"].tolist()

            # 3. Create a new column in df that maps to the closest correct name
            df_qcew["matched_name"] = df_qcew["name"].apply(
                lambda x: self.get_best_match(x, correct_names)
            )

            # 4. Merge df with df2 based on the new matched column
            result = pd.merge(
                df_qcew,
                gdf,
                left_on="matched_name",
                right_on="name",
                suffixes=("_original", "_cleaned"),
            )
            data = pl.DataFrame(result)
            data = data.with_columns(muni="name_cleaned")
            data = (
                data.group_by(["year", "month", "muni", "sector_code"])
                .agg(pl.col("employment").sum())
                .sort(["year", "month", "muni", "sector_code"])
            )
            data = self.data_trace(data.to_pandas())
            data = pl.DataFrame(data)
            data = data.group_by(["year", "month", "muni"]).agg(
                capital_synth=pl.col("synthetic_imports").sum(),
                labour=pl.col("employment").sum(),
            )
            energy = self.data_energy()
            energy = energy.with_columns(
                muni=(
                    pl.col("muni")
                    .str.normalize("NFD")
                    .str.replace_all(r"[\u0300-\u036f]", "")
                    .str.to_lowercase()
                    .str.replace_all("  ", "_")
                    .str.replace_all(" ", "_")
                )
            )
            data = data.join(
                energy, on=["year", "month", "muni"], how="inner", validate="1:1"
            )
            data.write_parquet(file_path)

            return data
        else:
            return pl.read_parquet(file_path)

    def data_trace(self, new_df):
        df_qcew = (
            self.data_qcew()
            .group_by(["year", "month", "sector_code"])
            .agg(pl.col("employment").sum())
        )
        df_imports = self.data_imports()
        df = df_qcew.join(
            df_imports, on=["year", "month", "sector_code"], how="inner", validate="1:1"
        ).to_pandas()

        time_coords = (
            df[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
        )
        time_coords["time_idx"] = range(len(time_coords))

        # Merge this index back into the main dataframe
        df = df.merge(time_coords, on=["year", "month"], how="left")

        # 2. Preprocessing categorical indexes
        df["sector_idx"], sectors = pd.factorize(df["sector_code"])

        n_sectors = len(sectors)
        n_periods = len(time_coords)  # Total number of months in the timeseries

        # Coordinate system for the model
        coords = {
            "time": range(n_periods),
            "sector": sectors,
            "fourier_features": ["sin", "cos"],
        }
        coords["obs_id"] = np.arange(len(df))
        month_scaled = 2 * np.pi * df["month"].values / 12
        sin_t = np.sin(month_scaled)
        cos_t = np.cos(month_scaled)

        with pm.Model(coords=coords) as import_model:
            # --- 1. Data Containers ---
            # Use ConstantData for arrays. This converts them to PyTensor types automatically.
            s_idx = pm.Data("s_idx", df["sector_idx"].values, dims="obs_id")
            t_idx = pm.Data("t_idx", df["time_idx"].values, dims="obs_id")
            log_emp = pm.Data(
                "log_emp", np.log(df["employment"].values + 1), dims="obs_id"
            )

            # --- 2. Sector-Level Random Walk (Non-Centered) ---
            state_drift_mu = pm.Normal("state_drift_mu", mu=0, sigma=0.01)
            sector_drift = pm.Normal(
                "sector_drift", mu=state_drift_mu, sigma=0.05, dims="sector"
            )
            sigma_rw = pm.Exponential("sigma_rw", 2.0)

            z_innovations = pm.Normal("z_innovations", 0, 1, dims=("sector", "time"))

            init_val = np.log(df.groupby("sector_idx")["imports"].first().values + 1)
            init_import = pm.Normal(
                "init_import", mu=init_val, sigma=0.5, dims="sector"
            )

            # Path calculation
            rw_component = (z_innovations * sigma_rw).cumsum(axis=1)
            # Using np.arange is fine here as it broadcasts against the tensor
            drift_component = sector_drift[:, None] * np.arange(n_periods)

            sector_path = pm.Deterministic(
                "sector_path",
                init_import[:, None] + drift_component + rw_component,
                dims=("sector", "time"),
            )

            # --- 3. Fourier Seasonality ---
            beta_fourier = pm.Normal(
                "beta_fourier", mu=0, sigma=0.5, dims=("sector", "fourier_features")
            )

            # Corrected: Ensure sin_t and cos_t are also treated as data/tensors
            s_t = pm.Data("sin_t", sin_t, dims="obs_id")
            c_t = pm.Data("cos_t", cos_t, dims="obs_id")

            seasonality = beta_fourier[s_idx, 0] * s_t + beta_fourier[s_idx, 1] * c_t

            # --- 5. Employment Relationship ---
            beta_employment = pm.HalfNormal("beta_employment", sigma=1.0)

            # --- 6. Expected Value ---
            mu = pm.Deterministic(
                "mu",
                sector_path[s_idx, t_idx] + (beta_employment * log_emp) + seasonality,
                dims="obs_id",
            )

            # --- 7. Likelihood ---
            sigma_obs = pm.Exponential("sigma_obs", 1.0)
            nu = pm.Exponential("nu", 0.1) + 1

            pm.StudentT(
                "obs",
                nu=nu,
                mu=mu,
                sigma=sigma_obs,
                observed=np.log(df["imports"].values + 1),
                dims="obs_id",
            )

            trace = pm.sample(1000, tune=1000, target_accept=0.95)

            new_df["sector_idx"] = pd.Categorical(
                new_df["sector_code"], categories=sectors
            ).codes

            # 2. Map Year/Month to the same time_idx used in training
            # time_coords is the dataframe with columns ['year', 'month', 'time_idx']
            new_df = new_df.merge(
                time_coords[["year", "month", "time_idx"]],
                on=["year", "month"],
                how="left",
            )

            # 3. Calculate Fourier features for the specific months in new_df
            new_month_scaled = 2 * np.pi * new_df["month"].values / 12
            new_sin_t = np.sin(new_month_scaled)
            new_cos_t = np.cos(new_month_scaled)

            # 4. Log-transform employment
            new_log_emp = np.log(new_df["employment"].values + 1)

        # 1. Check for missing indices after the merge
        if new_df["time_idx"].isnull().any():
            print(
                "Warning: Some rows in new_df don't have a matching time_idx. Dropping them."
            )
            new_df = new_df.dropna(subset=["time_idx", "sector_idx"])

        # 2. Prepare the values directly from existing columns
        # Make sure "employment" is the correct name of your column in new_df
        new_s_idx = new_df["sector_idx"].values.astype("int32")
        new_t_idx = new_df["time_idx"].values.astype("int32")

        # Calculate log(emp + 1) and force float64
        new_log_emp_values = np.log(new_df["employment"].values + 1).astype("float64")

        # Re-calculate Fourier features for the new months if you haven't yet
        new_month_scaled = 2 * np.pi * new_df["month"].values / 12
        new_sin_t = np.sin(new_month_scaled).astype("float64")
        new_cos_t = np.cos(new_month_scaled).astype("float64")

        with import_model:
            # 1. Update the data as before
            pm.set_data(
                new_data={
                    "s_idx": new_s_idx,
                    "t_idx": new_t_idx,
                    "log_emp": new_log_emp_values,
                    "sin_t": new_sin_t,
                    "cos_t": new_cos_t,
                },
                coords={"obs_id": np.arange(len(new_df))},
            )

            # 2. Add 'predictions=True' to avoid coordinate conflicts with training data
            post_pred = pm.sample_posterior_predictive(
                trace, var_names=["obs"], predictions=True
            )

        # 3. Extract from the 'predictions' group instead of 'posterior_predictive'
        synthetic_log_mean = (
            post_pred.predictions["obs"].mean(dim=["chain", "draw"]).values
        )

        # 4. Transform back
        new_df["synthetic_imports"] = np.exp(synthetic_log_mean) - 1

        return new_df

    def data_energy(self):
        for file in os.listdir(path="data/raw/"):
            if file[0:4] == "luma":
                year = int(file[5:-4])
                df = pl.read_csv(f"data/raw/{file}")
                df = df.unpivot(
                    on=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"],
                    index="muni",
                    value_name="energy",
                    variable_name="month",
                )
                df = df.with_columns(
                    month=pl.col("month").cast(pl.Int16),
                    energy=pl.col("energy").cast(pl.Float64),
                    year=pl.lit(year),
                )
                df.write_parquet(f"{self.saving_dir}raw/energy-{year}.parquet")

        data = self.conn.execute(
            f"SELECT * FROM '{self.saving_dir}raw/energy-*.parquet';"
        ).pl()
        return data

    def get_best_match(self, name, choices, threshold=80):
        # process.extractOne returns (match, score, index)
        match, score = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)

        # Only return the match if it's "good enough"
        return match if score >= threshold else None
