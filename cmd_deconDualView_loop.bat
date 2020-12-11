for /l %%x in (0, 1, 2) do (
   .\bin\win\deconDualView -i1 .\data\results_batch\RegA\SPIMA_reg_%%x.tif -i2 .\data\results_batch\RegB\SPIMB_reg_%%x.tif -fp1 .\data\PSFA.tif -fp2 .\data\PSFB.tif -o .\data\results\Decon_%%x.tif -it 10 -dev 0 -verbON  -cOFF
)