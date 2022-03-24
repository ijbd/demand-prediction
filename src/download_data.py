import pandas as pd
import os
import argparse

GH_URL = {"demand" : lambda bal_auth_code:
						'/'.join(["https://raw.githubusercontent.com",
						"truggles",
						"EIA_Cleaned_Hourly_Electricity_Demand_Data",
						"master",
						"data",
						"release_2020_Oct",
						"balancing_authorities",
						f"{bal_auth_code.upper()}.csv"]),
			"temperature" : lambda bal_auth_code :
						'/'.join(["https://raw.githubusercontent.com",
						"ijbd",
						"population_weighted_temperature",
						"main",
						"output",
						f"{bal_auth_code.lower()}-temperature-2020-pop.csv"]),
			"temperature-centroid" : lambda bal_auth_code :
						 '/'.join(["https://raw.githubusercontent.com",
						"ijbd",
						"population_weighted_temperature",
						"main",
						"output",
						f"{bal_auth_code.lower()}-temperature-centroid.csv"])}

def download_data(bal_auth_code: str, variable: str, output_file: str) -> None:
	''' Download data for a selected variable and save to file.'''
	
	assert variable in GH_URL, f"Invalid variable: {variable}."

	download_url = GH_URL[variable](bal_auth_code)
	
	# download from url
	data = pd.read_csv(download_url)

	# save to file
	data.to_csv(output_file, index=False)

	return None	

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("bal_auth_code", type=str, help="Balancing authority abbreviation code.")
	parser.add_argument("variable", type=str, choices=["demand", "temperature", "temperature-centroid"]) 
	parser.add_argument("output_file", type=str)
	args = parser.parse_args()
	
	download_data(args.bal_auth_code, args.variable, args.output_file)
