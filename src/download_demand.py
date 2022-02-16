import pandas as pd
import os
import argparse

def download_demand(bal_authority,raw_demand_filepath):
	'''
	Download hourly, balancing authority-scale demand and save to CSV.
	Data from https://github.com/truggles/EIA_Cleaned_Hourly_Electricity_Demand_Data

	:param bal_authority: Abbreviated balancing authority code.
	:type bal_authority: str
	:param raw_demand_filepath: Path to save demand data CSV.
	:type raw_demand_filepath: str
	:returns: None
	:rtype: None
	'''

	download_url = '/'.join(['https://raw.githubusercontent.com',
								'truggles',
								'EIA_Cleaned_Hourly_Electricity_Demand_Data',
								'master',
								'data',
								'release_2020_Oct',
								'balancing_authorities',
								'{}.csv'.format(bal_authority)])
	
	# download from url
	data = pd.read_csv(download_url)

	# save to file
	data.to_csv(raw_demand_filepath,index=False)

	return None

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('bal_authority',type=str,help='Balancing authority abbreviation code.')
	parser.add_argument('raw_demand_filepath',type=str)
	args = parser.parse_args()
	
	download_demand(args.bal_authority,args.raw_demand_filepath)
