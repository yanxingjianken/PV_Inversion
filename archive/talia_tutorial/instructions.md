How it works (theoretically): just like in the Davis paper. You take your observational or model data, and solve for PV. Then, before you do any science, you test the inversion by assuming some balance conditions, like hydrostatic, gradient wind balance, and then invert the PV field back to winds again. Now you compare the inverted winds to the original winds to see if this method is a good candidate for piecewise inversion. If the winds are kind of similar, i.e. if the inversion works and recovers a significant portion of the large scale flow, then it is a good tool to use.

THEN you start the piecewise process. You identify an anomaly, as some departure of a time mean. So you take the instantaneous field and subtract the time mean to get a perturbation state. I think it's this perturbation that you invert to get the winds.

It looks like a former student of Kerry's, C.C. Wu, last updated the code. A reference is at https://github.com/tbarbero/PPVI.

The program is 3 .f files. they may need to be updated slightly for modern compilers. The folder also holds example input and output files.

Chun Chieh Wu has added some very detailed program descriptions in the headers of the .f files.

1) Create a .grid file of your GCM data, containing geopotential, theta and velocity (u first then v) for each time step. The .grid data is the input to the fortran program. Output will be total PV field (not regular PV, but 2.3 from the Davis paper), like a 'dateXX_q.out' file. This PV does not assume QG balance, but allows for some gradient balance (Ro not necessarily small) as well. You also get a 'dateXX_h.out' file, I think there is where it converts from your geopotential to the preferred pseudoheight.

2) Now send this first output into the second program, pvpialln_94UV.f. This program should invert the q assuming some balance condition, which it computes on pseudoheight, and also compute the mean state. The mean state must be an average over at least ten days. Then the perturbation is just the deviation from the mean at each time step you're looking at (2.5 and 2.6 in the Davis paper).

3) Finally, you can invert the total PV with qinvert21_94.f, and each of the perturbation q with qinvertp21_94.f.
