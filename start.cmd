@echo off

rem List of years
set years=2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 2016 2017


rem Specify the directory where you want to download the data
set "download_dir=.\data\input_data"

rem Check if the directory exists, and create it if it doesn't
if not exist "%download_dir%" mkdir "%download_dir%"

rem Iterate through the list of years and download the data
for %%y in (%years%) do (
    rem Download the data for the current year
    powershell -Command "Invoke-WebRequest -Uri 'https://pds-ppi.igpp.ucla.edu/ditdos/write?id=urn:nasa:pds:cassini-mag-cal:data-1min-ksm:%%y_fgm_ksm_1m::1.0&f=csv' -OutFile '%download_dir%\%%y_FGM_KSM_1M.csv'"
    powershell -Command "Invoke-WebRequest -Uri 'https://pds-ppi.igpp.ucla.edu/ditdos/write?id=urn:nasa:pds:cassini-mag-cal:data-1min-krtp:%%y_fgm_krtp_1m::1.0&f=csv' -OutFile '%download_dir%\%%y_FGM_KRTP_1M.csv'"
