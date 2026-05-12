from jp_imports import JPTrade
import polars as pl
from jp_tools import download
from pathlib import Path
import geopandas as gpd
import tempfile
import hashlib
import pandas as pd
from thefuzz import process, fuzz


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

        if not file_path.exists():
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
            data = data.with_columns(name="name_cleaned")
            data = data.group_by(["year", "month", "sector_code", "qtr", "name"]).agg(
                pl.col("employment").sum()
            )
            data = data.join(
                self.data_imports(),
                on=["year", "month", "sector_code"],
                how="inner",
                validate="m:1",
            )
            data.write_parquet(file_path)
            return data
        else:
            return pl.read_parquet(file_path)

    def get_best_match(self, name, choices, threshold=80):
        # process.extractOne returns (match, score, index)
        match, score = process.extractOne(name, choices, scorer=fuzz.token_sort_ratio)

        # Only return the match if it's "good enough"
        return match if score >= threshold else None
