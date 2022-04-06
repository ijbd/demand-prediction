import pandas as pd
import argparse

def download_data(download_url: str, output_file: str) -> None:
	''' Download file from url and save to csv.'''
		
	# download from url
	data = pd.read_csv(download_url)

	# save to file
	data.to_csv(output_file, index=False)

	return None

if __name__ == "__main__":
	parser = argparse.ArgumentParser("Download csv data from URL.")
	parser.add_argument("download_url", type=str)
	parser.add_argument("output_file", type=str)

	args = parser.parse_args()
	
	download_data(args.download_url, args.output_file)