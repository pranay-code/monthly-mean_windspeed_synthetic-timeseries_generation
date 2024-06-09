# monthly-mean_windspeed_synthetic-timeseries_generation
INTRODUCTION:

The code helps you build or extrapolate the wind speed data to upto 1958-01 back in time with trade off being the quality (in comparison to provided observation data).

Total three time series (Synthetic) are built with time range 1958-01 to last month data in observation provided. Names of the three synthetic time series are RF, GBDT and LGBDT. 

Frequency: Monthly average

Supervised machine learning algorithms (RandomForest, XGBoostRegressor, LGBRegressor) are used for regressing reanalysis wind speed to observation wind speed.



SYSTEM REQUIREMENTS:

The following code is developed in ipython notebook using some of the external packages (not included in default python3) which are required
to be installed by the user.

STEPS:

1. Anaconda installation

	for all windows, linux and mac (please follow the instructions provided in the link below);
	
	https://docs.anaconda.com/anaconda/install/

	Also refer to the following link for quick start guide

	http://www.cdt-pv.org/media/resources/Anaconda-Quickstart.pdf

	NOTE: Python version installed should be >=3.7. 
	
	      verify this using the following command in terminal or command prompt.
	
	      python3 -V


2. Launch jupyter notebook

	for linux and mac users;

	once after installing the anaconda, open terminal and enter the following command.
	
	jupyter notebook
	
	On  successful installation of anaconda it will open a new tab in your browser.
	
	for windows users;
	
	please refer this video: https://www.youtube.com/watch?v=EbYGBANqDdY


3. Installing packages: 

	After successfully opening jupyter notebook, please terminate the jupyter notebook command
	and install the following package through the terminal or command prompt.

	(Windows, Linux, Mac)

        pip install -U scikit-learn==1.0.1	
        
        pip install pandas

        pip install xgboost
        
        pip install lightgbm

        pip install matplotlib

	NOTE: Since most of the packages such as pandas, numpy and matplotlib etc. are dependencies
	      to darts package they will be automatically installed.


After installation of following packages please restart (close and reopen) your terminal or command
prompt and open the jupyter notebook using following command.

	jupyter notebook


Once it opens, redirect to the forecast directory downloaded and click on the file named CODE.ipynb.
It opens the code file.


The following code is didvided into sections and sub-sections and with some code cells or snippets below each.
These code cells are to be executed (using run button on the toolbar above or shortcut key shift + enter) in sequence as arranged.



INPUT:


1. Observation data:

	Code requires wind speed observation data in the file named obs_data.csv placed in the same directory or folder.


	Data specifications:
        
        
        File format: CSV (comma seperated value)
        
        
		Columns: DATE (YYYY-MM-DD), windspeed (real number)
	
            	Example:
	
			DATE, windspeed
			2014-01-01, 0.6173328
			2014-01-02, 0.4115552
			2014-01-03, 0.9774436
			2014-01-04, 0.6173328
			2014-01-05, 0.771666
            
            
		Frequency: Daily average
		
		Length: Atleast two years i.e 2 * 365 data-points or rows excluding nan values.
		
	
	NOTE: Saving data using excel is changing DATE format to DD-MM-YYYY which  gives semantic error.  	




2. Reanalysis data:

	Code requires reanalysis data in the file named reanalysis_data.csv placed in the same directory or folder. Meteorological variables data sourced from reanalyses namely
	ERA5, JRA55 and NCEP1 of the nearest gridcell (to station location) are provided as features to correct the difference of reanalyses wind speeds from observation wind speed.
        
        Some of the features and their corresponding reanalysis dataset name are mentioned below.

        	wind_era5: Wind speed at 10 metres altitude

		wind_100_era5: Wind speed at 100 metres altitude

		wind_850hpa_era5: Wind speed at 850 hecta-pascal pressure level

		bld_era5: Boundary layer depth

		blh_era5: Boundary layer height

		t2m_era5: Air temperature at 2 metres above the ground

		d2m_era5: Dew-point temperature at 2 metres above the ground

		sshf_era5: Sensible heat flux

		wind_jra55: Wind speed at 10 metres altitude

		T2m_jra55: Air temperature at 2 metres above the ground

		SH_jra55: Sensible heat flux

		LH_jra55: latent heat flux

		surf_roughness_jra55: Surface roughness

		wind_ncep1: Wind speed at 10 metres altitude
	
	Compulsory features: wind_era5, wind_jra55 and wind_ncep1 (names should be same)
	        
        Minimum number of features required: 10
        
        Maximum number of features required: NO LIMIT

	NOTE: There is no limit to the maximum number of features. Higher the number of relevent features could provide better synthetic timeseries but trade off is time for running code increases.  	


        File format: CSV (comma seperated value)
         

	Frequency: Daily average
		
	
        Length: Atleast two years i.e 2 * 365 data-points(with non nan values).
		
	Example:
		DATE,wind_era5,wind_100_era5,wind_850hpa_era5,bld_era5,blh_era5,t2m_era5,d2m_era5,sshf_era5,wind_jra55,T2m_jra55,SH_jra55,LH_jra55,surf_roughness_jra55,wind_ncep1
		1958-01-01,2.0614414,3.3415825,8.150538,4355.6665,208.04854,290.58347,285.30124,-64983.09,0.71065634,291.2378,5.960682,12.766289,0.84573656,4.3611183
		1958-01-02,2.3030872,4.7519317,7.645471,4154.268,226.9663,292.01657,285.8823,-35037.184,1.1201906,292.56107,5.5807114,16.348362,0.84573656,3.671173
		1958-01-03,2.1578903,4.5576844,8.323084,3699.5547,290.52725,292.1628,285.45242,-72491.305,0.9845711,292.70026,7.4428062,16.067474,0.84573656,3.3398626
		1958-01-04,2.5109131,4.899772,6.158053,2438.1667,301.6722,290.4771,284.4077,-45679.902,1.2660966,290.73218,9.007015,15.497513,0.84573656,1.8862176	
								:
								:
								:
								:
								:
								:
								:
		2022-02-24,2.1327965,3.4818623,2.6037252,2676.8164,319.70438,294.38095,289.10574,-94237.43,1.0164527,294.931,28.07164,45.133583,0.80960375,1.820503
		2022-02-25,2.1462681,3.8101218,4.6625366,2370.6238,570.69403,293.6689,287.16483,-108775.02,0.4664938,294.85425,33.53095,46.30238,0.80960375,2.0941987
		2022-02-26,3.2053602,5.8540363,4.477964,5167.4087,547.5441,294.07816,287.56082,-88454.125,1.1041417,294.26648,32.675194,51.176567,0.80960375,3.9115973
		2022-02-27,2.7507718,4.712565,4.464536,3729.5312,364.13834,293.77148,287.33823,-121414.375,1.1026266,295.26263,31.042133,51.15106,0.80960375,2.9249058
		2022-02-28,2.879447,4.80146,7.2892003,5216.819,419.17422,292.0748,284.2382,-143493.64,1.145201,293.61102,50.85142,68.00644,0.80960375,3.5035574


	Time range: 1958-01-01 to end of observation time series. 

	NOTE: Saving data using excel is changing DATE format to DD-MM-YYYY which  gives semantic error.  	
