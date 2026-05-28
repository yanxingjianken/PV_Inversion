C From:	IN%"CDAVIS@MMM.UCAR.EDU" 18-SEP-1991 17:56:42.42
C To:	WU@CMPO.MIT.EDU
C Subj:	DOCUMENTATION

C Date: Wed, 18 Sep 1991 11:56:03 MDT
C From: CDAVIS@MMM.UCAR.EDU
C Subject: DOCUMENTATION
C To: WU@CMPO.MIT.EDU
C Message-Id: <910918115603.671f@MMM.UCAR.EDU>
C X-Vmsmail-To: SMTP%"WU@CMPO.MIT.EDU"

	PROGRAM QINVERT
C*****************************************************************
C        This version of QINVERT contains the long-awaited documentation
C        which I hope will finally settle the issue of what the Hell 
C        is actually going on in this program. As with the inversion
C        process itself, many iterations of this documentation
C        will probably be required before QINVERT is decribed
C        well enough to be used.
C**************************************************
C        There is one Subroutine: BALNC.  This is where all
C        the action occurs.  The main body of the program is devoted 
C        to reading in the initial guess and PV fields, setting
C        some of the differential operator coefficients and
C        nondimensionalizing variables. The main body also
C        writes out the balanced streamfunction (hereafter SI in the main
C        program, S in BALNC) and geopotential (HZ in main; H
C        in BALNC.
C***********************************
C        Variables: Main Program
C***********************************
C        NL= # of levels. These pressure levels must be specified
C            in a DATA statement. Level NL is the top of the
C            domain. SI and HZ exist at all these
C            levels; there is no PV defined at levels 1 and NL.
C        NX= # of grid points in 'x' or longitudinal direction.
C            X increases to the east.
C        NY= # of grid points in Y or latitude. Y increases to the
C            SOUTH. The reasons for this bizarre convention remain
C            obscure, but i=NL is along the sourthern boundary of
C            domain.
C     NOTE:: IF you change NY,NX,NL here, you must do so in BALNC, too.
C        Q(i,j,k)=PV; This variable is read in. Expected units are
C             0.01 PVU, i.e. PV in the input file should have values
C             of order 100.
C        THZ(i,j,k)= Potential temperature (theta) at half levels (there
C            are really NL-1 levels of this). THZ is only used if
C            your initial guess is in terms of theta. This option is
C            specified by INF, to be explained in the OPTIONS section
C            below.
C        AP(i)=cos(latlitude(i)) if you are in lat/lon coords. Otherwise
C            it is set to 1.
C        APP(i)=cos(latitude(i+1/2)). APP(i)>AP(i)
C        APM(i)=cos(latitude(i-1/2)). APM(i)<AP(i)
C        HZ(i,j,k)=geopotential height. Once nondimensionalized, this
C            becomes geopotential. Units of input HZ are meters.
C        SI(i,j,k)=streamfunction. Units of input are 10**5 m**2/s, 
C            i.e., the numbers for input PSI should look like
C            geopotential heights in meters.
C        ZHDR(m)=Header array at the top of each file. This
C            is mostly for plotting purposes. It contains the 
C            grid size and horizontal resolution (degrees lat if in
C            lat/lon or in km if in x/y coords--- HDR(5) is long
C            resolution; HDR(6) is latitude resol.). HDR(7)=NX
C            and HDR(8)=NY.
C        PI(k) is not 3.1415926535........, but rather the Exner function
C            (which is really the vertical coordinate used).
C        THT(i,j,m)= Boundary theta, m=1 is lower boundary (midway
C            between k=1 and k=2) and m=2 is upper boundary
C            (between k=NL-1 and k=NL).
C        PIF(k)=basically the inverse Pseudo-density, something
C            like GG*KAP*PI(k)/PR(k). I simply normalize Q by this and 
C            the mysterious..
C        QCONST, which is the parameter that pops out of the
C            nondimensionalization.
C        FC(i,j)= Coriolis parameter. It is 2-D in case one is
C            using map-factor coordinates, in which case 'y' and latitude
C            do not correspond.
C        PR(k)=Pressure (BARS) specified by DATA statement.
C        THZB(k)=area average of THT
C        A(i,m)=Coefficients of the 2-D Laplacian operator (Standard
C            5 point arrangement about central grid point i,j,k:
C            m=1 is one point north, m=2 is one point west, m=4
C            is one point east and m=5 is one point south).
C        MF(i,j)=Map factor. This is set to 1 in lat/lon coords.
C        FM(i,j)=FC/(MF*MF)  Big deal.
C        Latt(i,j)=latitude of grid points (only needed in map-factor
C            coords.)
C        QDIF(NL)=Amount of PV that is subtracted from all values
C            to preserve volume integral of PV if there are any points
C            in the domain with negative PV. The inversion will
C            not converge if there are any negative PV values
C            in the domain. Therefore, any that exist have to be set
C            to a small, positive value (QMIN). Because the PV increases
C            at some points, we subtract a very tiny amount (QDIF)
C            at all other points to preserve the volume integral. In
C            practice, this is no big deal.
C        OMEGAS=relaxation parameter for SI, which is solved for
C            in BALNC in a 2-D Poisson eq. IF your grid is 20x20,
C            try 1.6 or so. IF it's 40x40 try 1.85-1.9.
C        OMEGAH=relaxation parameter for 3-D Poisson eq. solved
C            for HZ. Usually the same as OMEGAS, but it can help
C            to make it slightly smaller.
C        FF,LL,THO,DPI and FRC(=1) are all constants used
C            in nondimensionalization.
CC****************************************************
C       OPTIONS:
C            To understand these options, first it may help to have an 
C            overview of how the inversion works. The equations
C            we are trying to solve are 
C
C             delsq(H)=div(f grad(S)) + 2(S(xx)S(yy) - S(xy)S(xy)) (1)
C             q=(f + delsq(S))H(zz) - S(xz)H(xz) - S(yz)H(yz)      (2)
C
C            where, with this crazy notation (x) denotes a partial
C            derivative w.r.t. x, etc.
C
C            The first step is to add (1) and (2).  This gives an eq.
C            for S
C
C             delsq(S(new))=Stuff( H(new), q, S(old)), where S(old))    (3)
C            
C            where S(old) is the previous solution (or initial guess)
C            for S and is used to evaluate the nonlinear term. This
C           is the 2-D Poisson eq. alluded to above.
C           IF we subtract (2) from (1) it gives a 3-D eq. for H
C            
C            delsq(H) + (f + delsq(S(new)))H(zz) = Stuff( H(old),
C                    S(new), q)                                       (4)
C
C            We start with (4), and use SOR to iterate until H is changing
C            by a sufficiently small amount everywhere (specified
C            by THR (in meters, usually 0.1)). Then we underralax
C            to get the vbalue of H(new) which will be stuck into
C            the r.h.s. of (3)
C
C              H(new) = PRT*H(new) + (1.-PRT)*H(old). This is where
C            
C            PRT comes in.  Then we iterate on (3) until the same 
C            convergence level is achieved. We underrelax S as we
C            did with H to get a new S that we stick into (4).  The
C            process is repeated until both (3) and (4) converge in
C            one iteration.
CC*******************************************
C           With all that in mind, then...
C        MAX=maximum allowed iteration for either (3) or (4). If convergence
C            doesn't happen, the fields are written out and the program ends.
C            It's good to have MAX at least 100.
C        MAXT=maximum number of cycles through both (3) and (4). Should
C            be set to at least 30 or 40.
C         IMAP= option for lat/lon (=1) or x/y (=2) (map factor coords).
C             if IMAP=2, then you must also have a file with the
C             map-factor values and latitudes of each grid point. Don't
C             forget about the file header.
C         INF= option for type of initial guess. IF INF=1, it is
C             assumed that the INIT file has NL levels of HZ,
C             followed by NL levels of SI as the initial guess.  IF
C             INF.ne.1 the INIT file must contain NL-1 levels of theta
C             followed by SI at the same (NL-2) levels where PV
C             is defined. To get the initial guess for HZ, one also
C             must supply HZ at level NL, and then a downward
C             integration of the hydrostatic equation is performed.
C             It may be easier at first to set INF=1 and come up
C             with your own initial guess for the H and S fields.
C             One possibility is to set S=H iniitally, which is
C             the equivalent of geostrophic lateral boundary
C             conditions. H can be obtained directly from the data.
C****************************************************************
C          So how fast does this run?
C
C           With the grid size you see dimensioned below, this program
C           should take about 6-8 minutes on the VAX 3500.
C
C          The numbers in the output file have exactly the
C          same units as the numbers in the input file.
C******************************************************************
	PARAMETER (NL=10)
        PARAMETER (NY=51)
        PARAMETER (NX=87)
	REAL Q(NY,NX,NL),THZ(NY,NX,NL),AP(NY),APP(NY),APM(NY),
     +       HZ(NY,NX,NL),THT(NY,NX,2),ZHDR(8),PI(NL),KAP,PIF(NL),
     +       FC(NY,NX),SIGM,AA,BETA,SI(NY,NX,NL),PR(NL),P0,THZB(NL),
     +	     R,CP,CV,FF,UC,LL,PII,MI,FRC,A(NY,5),PRT,QMIN,THR,
     +       MF(NY,NX),FM(NY,NX),LATT(NY,NX),QDIF(NL),QNEW,QCONST,
     +       OMEGAS,OMEGAH,GG,THO,DPI
        INTEGER QL(NL),MAX,MAXT
	CHARACTER*40 FNM(10),LLF
	DATA PR/ 1., .925, .85, .7, .6, .5, .4, .3, .25, .2/
C        DATA PR/ 1., 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6,
C     +       0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15 /
	PRINT*,'MAX ITER (IND AND TOT), OMEGAS, OMEGAH, PRT, THRSH ?'
	READ(5,*)MAX
	READ(5,*)MAXT
	READ(5,*)OMEGAS
        READ(5,*)OMEGAH
	READ(5,*)PRT
	READ(5,*)THR
        WRITE(6,*)MAX,MAXT,OMEGAS,OMEGAH,PRT,THR
	PRINT*,'FILE NAMES FOR INIT,Q (INPUT) AND BAL (OUTPUT)?'
        READ(5,*)FNM(1)
        READ(5,*)FNM(2)
        READ(5,*)FNM(5)
	DO 199 KL=1,2
	 OPEN(KL+6,FILE=FNM(KL),STATUS='OLD')
	 READ(KL+6,*)(ZHDR(I),I=1,8)
 199    CONTINUE
	OPEN(11,FILE=FNM(5),STATUS='new')
	WRITE(11,*)(ZHDR(I),I=1,8)
        PRINT*,'ENTER 1 FOR SPHERICAL, 2 FOR XY PROJ.'
        READ(5,*)IMAP
C******** SET CONSTANTS ************
  9	PII=4.*ATAN(1.)
	DPI=500./FLOAT(NL)
        PRINT*,'ENTER QMIN (PVU)'
        READ(5,*)QMIN 	
	AA=2.E7/PII
	CP=1004.5
	R=2.*CP/7.
	CV=CP-R
	FF=1.E-4
	KAP=R/CP
	P0=1.E5
	MI=9999.90
	GG=9.81
	SIGM=ZHDR(5)/ZHDR(6)	
        IF (IMAP.EQ.1) THEN
	 LL=AA*ZHDR(5)*PII/180.
         DO 101 I=1,NY
         AP(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-1.)*ZHDR(6))/180. )
	 APM(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-1.5)*ZHDR(6))/180. )
	 APP(I)=COS( PII*(ZHDR(3) - (FLOAT(I)-0.5)*ZHDR(6))/180. )
  	 A(I,1)=SIGM*SIGM*APM(I)/AP(I)
	 A(I,2)=1./( AP(I)*AP(I) )
	 A(I,3)=-( 2. + SIGM*SIGM*AP(I)*(APM(I)+APP(I)) )/
     +	    ( AP(I)*AP(I) )
	 A(I,4)=1./( AP(I)*AP(I) )
	 A(I,5)=SIGM*SIGM*APP(I)/AP(I)
         DO 104 J=1,NX
          FC(I,J)=1.458*SIN( PII*(ZHDR(3) - (FLOAT(I)-1)*ZHDR(6))/180. )
          MF(I,J)=1.
          FM(I,J)=FC(I,J)/MF(I,J)
 104     CONTINUE
 101     CONTINUE
        ELSE
         LL=ZHDR(5)*1.E3
         PRINT*,'LAT/LON FILENAME?'
         READ(5,*)LLF
         OPEN(23,FILE=LLF,STATUS='OLD')
         READ(23,*)(ZHDR(I),I=1,8)
         DO 105 I=1,NY
          READ(23,*)(MF(I,J),J=1,NX)
 105     CONTINUE        
         DO 103 I=1,NY
          READ(23,*,END=103)(LATT(I,J),J=1,NX)
 103     CONTINUE        
         DO 102 I=1,NY
         AP(I)=1.
  	 A(I,1)=SIGM*SIGM
	 A(I,2)=1.
	 A(I,3)=-2*( 1. + SIGM*SIGM )
	 A(I,4)=1.
	 A(I,5)=SIGM*SIGM
         DO 106 J=1,NX
          FC(I,J)=1.458*SIN( PII*LATT(I,J)/180. )
          MF(I,J)=MF(I,J)*MF(I,J)
          FM(I,J)=FC(I,J)/MF(I,J)
 106     CONTINUE
 102     CONTINUE
        END IF
	THO=FF*FF*LL*LL/DPI
	FRC=DPI*THO/(FF*FF*LL*LL)
	QCONST=1.E6*KAP*GG*CP*FF*THO/(P0*DPI)
	UC=DPI*THO/(FF*LL)
	BETA=1.458*LL/AA
        THR=GG*THR/(DPI*THO)
	WRITE(6,*)FRC,THO,DPI,QCONST,THR
	DO 202 K=1,NL
	 PI(K)=CP*( PR(K)**KAP )
	 PIF(K)=(PI(K)/CP)**2.5
	 PI(K)=PI(K)/DPI
 202    CONTINUE
C********** Read in hght data for initial guess. *********
        PRINT*,'INIT IS HGHT/PSI ("1") OR THTA/PSI ("2")?'
        READ(5,*)INF
        IF (INF.EQ.1) THEN
         DO 221 K=1,NL
         THZB(K)=0.
	 DO 221 I=1,NY
	  READ(7,*)(HZ(I,J,K),J=1,NX)
 	  DO 223 J=1,NX
	   HZ(I,J,K)=HZ(I,J,K)*GG/(THO*DPI)
 223      CONTINUE	
 221     CONTINUE
	 DO 286 K=1,NL
	 DO 286 I=1,NY
	  READ(7,*,END=288)(SI(I,J,K),J=1,NX)
 288 	  DO 287 J=1,NX
	   SI(I,J,K)=SI(I,J,K)*GG/(THO*DPI)
 287      CONTINUE	
 286     CONTINUE
C*************************************************	
        ELSE
         PRINT*,'ENTER FILENAME FOR UPPER B.C. HGHT.'
         READ(5,*)FNM(3)
         OPEN(9,FILE=FNM(3),STATUS='OLD')
         READ(9,*)(ZHDR(I),I=1,8)
         DO 241 K=1,NL-1
	 DO 241 I=1,NY
	  READ(7,*,END=11)(THZ(I,J,K),J=1,NX)
 11	  DO 243 J=1,NX
	   THZ(I,J,K)=THZ(I,J,K)/THO
           THZB(K)=THZB(K) + THZ(I,J,K)
 243      CONTINUE	
 241     CONTINUE
         THZB(1)=THZB(1)/FLOAT(NY*NX)	
         THZB(NL-1)=THZB(NL-1)/FLOAT(NY*NX)	
         WRITE(6,*)THZB(1),THZB(NL-1)
         DO 242 I=1,NY
	  READ(9,*,END=12)(HZ(I,J,NL),J=1,NX)
 12       DO 244 J=1,NX
           HZ(I,J,NL)=HZ(I,J,NL)*GG/(THO*DPI)
 244      CONTINUE
 242     CONTINUE
         DO 247 J=1,NX
         DO 247 I=1,NY
          DO 248 K=NL-1,1,-1
           HZ(I,J,K)=HZ(I,J,K+1) - THZ(I,J,K)*(PI(K)-PI(K+1))
 248      CONTINUE
 247     CONTINUE
         DO 290 K=2,NL-1
         DO 290 I=1,NY
	  READ(7,*,END=18)(SI(I,J,K),J=1,NX)
 18 	  DO 292 J=1,NX
	   SI(I,J,K)=SI(I,J,K)*GG/(THO*DPI)
           IF (K.EQ.NL-1) THEN
            SI(I,J,1)=SI(I,J,2) - (THZ(I,J,1)-THZB(1))*(PI(1)-PI(2))
            SI(I,J,NL)=SI(I,J,NL-1) + (THZ(I,J,NL-1)-THZB(NL-1))*
     +            (PI(NL-1)-PI(NL))
           END IF
 292      CONTINUE	
 290     CONTINUE
C*************************************************	
        END IF
C***********************************************************
 	 DO 251 K=1,2
	  DO 252 I=1,NY
	   READ(8,*)(THT(I,J,K),J=1,NX)
	   DO 253 J=1,NX
	    THT(I,J,K)=THT(I,J,K)/THO
 253       CONTINUE	
 252      CONTINUE	
 251     CONTINUE	
 	 DO 261 K=2,NL-1
	  DO 262 I=1,NY
	   READ(8,*,END=31)(Q(I,J,K),J=1,NX)
 262       CONTINUE	
 261       CONTINUE	
C************** Nondimensionaize PV ******************************
 31 	DO 271 K=2,NL-1
         QL(K)=0
         QDIF(K)=0.
         QNEW=PIF(K)*QMIN/QCONST
         write(6,*)qnew  
 	 DO 272 J=1,NX
	 DO 273 I=1,NY
	  IF (Q(I,J,K).NE.MI) THEN
	   Q(I,J,K)=PIF(K)*Q(I,J,K)/(MF(I,J)*1.E2*QCONST)      
C                                               !mult by d(p)/d(pi)
	   IF (Q(I,J,K).LE.QNEW) THEN
            QDIF(K)=QDIF(K) + QNEW - Q(I,J,K)
            QL(K)=QL(K) + 1
            Q(I,J,K)=QNEW
	   END IF
	  END IF
 273      CONTINUE	
 272     CONTINUE
         QDIF(K)=QDIF(K)/( FLOAT(NY*NX - QL(K)) )	
         DO 275 J=1,NX
         DO 276 I=1,NY
          IF (Q(I,J,K).GT.QNEW+QDIF(K)) THEN
           Q(I,J,K)=Q(I,J,K) - QDIF(K)
          END IF
 276     CONTINUE
 275     CONTINUE
 271    CONTINUE
        WRITE(6,109)(QL(K),K=2,NL-1)
        WRITE(6,1092)(QDIF(K),K=2,NL-1)
 109    FORMAT(10I8)
 1092   FORMAT(10F9.3)
C
        write(6,*)(hz(20,30,k),k=1,nl)
        write(6,*)(si(20,30,k),k=1,nl)
        WRITE(6,*)
C*********** ROUTINE TO SOLVE FOR H AND PSI BY SOR ***********************
C
	CALL BALNC(FC,FM,MF,AP,A,HZ,SI,Q,THT,SIGM,PI,PRT,THO,
     +	  THZB,THR,BETA,FRC,MAX,MAXT,QCONST,OMEGAS,OMEGAH)
C
C********** WRITE OUT BALANCED FIELDS ***********************
C
	DO 501 K=1,NL
	DO 502 I=1,NY
	 DO 503 J=1,NX
	  HZ(I,J,K)=HZ(I,J,K)*DPI*THO/GG
 503     CONTINUE
         WRITE(11,991)(HZ(I,J,K),J=1,NX)
 502    CONTINUE
 501    CONTINUE
	DO 505 K=1,NL
	 DO 506 I=1,NY
	  DO 507 J=1,NX
	   SI(I,J,K)=SI(I,J,K)*DPI*THO/GG
 507      CONTINUE
 	   WRITE(11,991)(SI(I,J,K),J=1,NX)
 506     CONTINUE	
 505    CONTINUE	
C************* END OF WRITING OUT *****************************
C
 991	 FORMAT(13f10.2)
C
	STOP
	END
C*******************************************************************
	SUBROUTINE BALNC(FCO,FCM,MFC,APS,AC,H,S,QE,THA,SIG,PE,
     +	 PART,THSC,THSB,THRS,BET,FR,MAXX,MAXXT,QCON,OMEGS,OMEGH)
	PARAMETER (NL=10)
        PARAMETER (NY=51)
        PARAMETER (NX=87)
	REAL  H(NY,NX,NL),QE(NY,NX,NL),SIG,S(NY,NX,NL),
     +    MI,ZM,THA(NY,NX,2),RH(NY,NX,NL),AC(NY,5),PE(NL),
     +	  ZPL(NY),ZPP(NY),BET,FR,FCO(NY,NX),MFC(NY,NX),THSB(NL),
     +	  APS(NY),RS,GPTS,STB(NY,NX,NL),ASI(NY,NX,NL),FCM(NY,NX),
     +	  BB(NL),BH(NL),BL(NL),VOR,PART,OLD(NY,NX,NL),THRS,THSC,
     +	  DPI2(NL),NLCO(NY,NX),COEF(NY,NL),DH(NY,NX,NL),QCON,
     +	  ZNL,RHS(NY,NX,NL),DELH(NY,NX,NL),DSI(NY,NX,NL),
     +    OMEGS,OMEGH
        INTEGER MAXX,MAXXT
	LOGICAL IT,ICON
        PRINT*,'ENTERED SUB. BALNC'
	MI=9999.90
	GPTS=FLOAT(NY*NX*(NL-2))
	WRITE(6,*)SIG,NL,GPTS
        ILTZ=0
        IGTZ=0
        INSQ=0
        ITCT=0
        ITC1=0
        ITC=0
        IITOT=0
        VOZRO=0.
        ITCOLD=0.
        DSIMAX=100.
	DO 201 I=2,NY-1
	 ZPL(I)=FR/(16.*APS(I)*APS(I))
	 ZPP(I)=FR*SIG*SIG/16.
         DO 2011 J=2,NX-1
	  NLCO(I,J)=2.*FR*MFC(I,J)*SIG*SIG/( APS(I)*APS(I) )
 2011    CONTINUE
 201    CONTINUE

	DO 202 K=2,NL-1
	 BB(K)=-2./( (PE(K+1)-PE(K))*(PE(K)-PE(K-1)) )
	 BH(K)=2./( (PE(K+1)-PE(K))*(PE(K+1)-PE(K-1)) )
	 BL(K)=2./( (PE(K)-PE(K-1))*(PE(K+1)-PE(K-1)) )
	 DPI2(K)=(PE(K+1) - PE(K-1))/2.
	 DO 203 I=1,NY
	  COEF(I,K)=AC(I,3)/BB(K)
 203     CONTINUE
 202    CONTINUE
        DO 204 K=1,NL
         DO 205 J=1,NX
         DO 206 I=1,NY
          OLD(I,J,K)=H(I,J,K)
 206     CONTINUE
 205     CONTINUE
 204    CONTINUE

C***********************************************

 900	CONTINUE		!TOTAL ITERATION 
	IF (IITOT.EQ.0) GOTO 700
	ITCM=0
	ITC1=0
	SPZRO=0
	DO 210 K=2,NL-1
	DO 211 J=2,NX-1
	DO 212 I=2,NY-1
	 OLD(I,J,K)=S(I,J,K) 	!PREVIOUS GUESS
         IF (K.EQ.2) THEN
          OLD(I,J,1)=S(I,J,1)
         ELSE IF (K.EQ.NL-1) THEN
          OLD(I,J,NL)=S(I,J,NL)
         END IF
	 DELH(I,J,K)= AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +	    AC(I,3)*H(I,J,K) + AC(I,4)*H(I,J+1,K) +
     +	    AC(I,5)*H(I+1,J,K)
         STB(I,J,K)=BL(K)*H(I,J,K-1) + BH(K)*H(I,J,K+1) + 
     +     BB(K)*H(I,J,K)
         IF (STB(I,J,K).LE.0.0001) THEN
          STB(I,J,K)=0.0001
          SPZRO=SPZRO + 1
         END IF
	 SXX=S(I,J+1,K)+S(I,J-1,K)-2.*S(I,J,K)
	 SYY=S(I-1,J,K)+S(I+1,J,K)-2.*S(I,J,K)
	 SXY=( S(I-1,J+1,K)-S(I-1,J-1,K)-
     +		S(I+1,J+1,K)+S(I+1,J-1,K) )/4.
         BETAS=0.25*( SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +      (S(I-1,J,K)-S(I+1,J,K)) + (FCO(I,J+1)-FCO(I,J-1))*
     +      (S(I,J+1,K)-S(I,J-1,K)) ) 
         ZHP=H(I-1,J,K+1)-H(I+1,J,K+1)-H(I-1,J,K-1)+H(I+1,J,K-1)
	 ZHL=H(I,J+1,K+1)-H(I,J-1,K+1)-H(I,J+1,K-1)+H(I,J-1,K-1)
         ZSP=S(I-1,J,K+1)-S(I+1,J,K+1)-S(I-1,J,K-1)+S(I+1,J,K-1)
	 ZSL=S(I,J+1,K+1)-S(I,J-1,K+1)-S(I,J+1,K-1)+S(I,J-1,K-1)
	 ZL=ZPL(I)*ZHL*ZSL/(DPI2(K)*DPI2(K))
	 ZP=ZPP(I)*ZHP*ZSP/(DPI2(K)*DPI2(K))
	 ZNL=NLCO(I,J)*( SXX*SYY - SXY*SXY ) + BETAS
         RHST=QE(I,J,K) - FCM(I,J)*STB(I,J,K) + DELH(I,J,K) -
     +      ZNL + ZL + ZP
         RHS(I,J,K)=RHST/(FCO(I,J) + FR*STB(I,J,K))
C********************************************************
 212    CONTINUE
 211    CONTINUE
 210    CONTINUE
	WRITE(6,23)SPZRO
 23	FORMAT(F6.0,' NEG STABILITIES.')
	SPZRO=0.
C************* ITERATION FOR PSI **********************
	DO 220 K=2,NL-1
	ITC=0
 800 	IT=.TRUE.
	 DO 221 J=2,NX-1
	 DO 222 I=2,NY-1
	  RS=AC(I,1)*S(I-1,J,K) + AC(I,2)*S(I,J-1,K) +
     +	    AC(I,3)*S(I,J,K) + AC(I,4)*S(I,J+1,K) +
     +	    AC(I,5)*S(I+1,J,K) - RHS(I,J,K)
	  DSI(I,J,K)=-OMEGS*RS/AC(I,3)
	  S(I,J,K) = S(I,J,K) + DSI(I,J,K)
C******* Check accuracy criterion *******************************
 	  IF (ABS(DSI(I,J,K)).GT.THRS) THEN
	   IT=.FALSE.
	  END IF
 222    CONTINUE
 221    CONTINUE

	ITC=ITC+1
	IF (IT) THEN
	 ICON=.TRUE.
	 IF (ITC.GT.ITCM) ITCM=ITC
	 IF (ITC.EQ.1) ITC1=ITC1 + 1
	 WRITE(6,*)ITC
	ELSE 
	 IF (ITC.LT.MAXX) THEN
	  GO TO 800
	 ELSE
	  PRINT*,'TOO MANY ITERATIONS FOR PSI.'
	  ICON=.TRUE.
	 END IF
	END IF

 802	CONTINUE
 220    CONTINUE

	PRINT*,'PSI CONVERGED.'
	IF (IITOT.GT.0) THEN
	ITCT=ITCT + ITCM
	DO 230 K=1,NL
	 DO 231 J=2,NX-1
	 DO 232 I=2,NY-1
	  S(I,J,K)=PART*S(I,J,K) + (1.-PART)*OLD(I,J,K)
	  OLD(I,J,K)=H(I,J,K)
 232     CONTINUE
 231     CONTINUE
 230    CONTINUE
	END IF
C************* CALCULATE THE RHS OF BAL+PV EQUATION ************
 700	DO 240 K=2,NL-1
	 DO 241 J=2,NX-1
	 DO 242 I=2,NY-1
	  VOR=AC(I,1)*S(I-1,J,K) + AC(I,2)*S(I,J-1,K) +
     +	    AC(I,3)*S(I,J,K) + AC(I,4)*S(I,J+1,K) +
     +	    AC(I,5)*S(I+1,J,K)
	  IF (VOR*FR.LE.0.0001-FCM(I,J)) THEN
	   VOR=(0.0001 - FCM(I,J))/FR
	   VOZRO=VOZRO + 1
	  END IF
	  ASI(I,J,K)=FCM(I,J) + FR*VOR
	  SXX=S(I,J+1,K)+S(I,J-1,K)-2.*S(I,J,K)
	  SYY=S(I-1,J,K)+S(I+1,J,K)-2.*S(I,J,K)
	  SXY=( S(I-1,J+1,K)-S(I-1,J-1,K)-S(I+1,J+1,K)+
     +        S(I+1,J-1,K) )/4.
          ZHP=H(I-1,J,K+1)-H(I+1,J,K+1)-H(I-1,J,K-1)+H(I+1,J,K-1)
	  ZHL=H(I,J+1,K+1)-H(I,J-1,K+1)-H(I,J+1,K-1)+H(I,J-1,K-1)
          ZSP=S(I-1,J,K+1)-S(I+1,J,K+1)-S(I-1,J,K-1)+S(I+1,J,K-1)
	  ZSL=S(I,J+1,K+1)-S(I,J-1,K+1)-S(I,J+1,K-1)+S(I,J-1,K-1)
	  ZL=ZPL(I)*ZHL*ZSL/(DPI2(K)*DPI2(K))
	  ZP=ZPP(I)*ZHP*ZSP/(DPI2(K)*DPI2(K))
          BETAS=0.25*( SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +      (S(I-1,J,K)-S(I+1,J,K)) + (FCO(I,J+1)-FCO(I,J-1))*
     +      (S(I,J+1,K)-S(I,J-1,K)) ) 
	  RHA=FCO(I,J)*VOR + NLCO(I,J)*(SXX*SYY - SXY*SXY) + BETAS
	  RH(I,J,K)=RHA + QE(I,J,K) + ZL + ZP
 242    CONTINUE
 241    CONTINUE
 240    CONTINUE

	WRITE(6,24)VOZRO
 24	FORMAT(F6.0,' NEG ABS VORTICITIES IN PHI EQ.')
	VOZRO=0.

C************* SOLVE FOR H AT EACH LEVEL *****************

 	ITC=0
 701 	IT=.TRUE.
        ZMRS=0.
	DO 250 K=2,NL-1
	DO 251 J=2,NX-1
	DO 252 I=2,NY-1
	 IF (K.EQ.2) THEN
	  RS=AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +	   ( AC(I,3) + ASI(I,J,K)*(BB(K)+BL(K)) )*H(I,J,K) + 
     +	    AC(I,4)*H(I,J+1,K) + AC(I,5)*H(I+1,J,K) + 
     +	    ASI(I,J,K)*(BH(K)*H(I,J,K+1) + THA(I,J,1)/DPI2(K))
     +	    - RH(I,J,K)
	  ZM=H(I,J,K)
	  H(I,J,K)=ZM - OMEGH*RS/(AC(I,3) + ASI(I,J,K)*(BB(K)+BL(K)))

	 ELSE IF (K.EQ.NL-1) THEN
	  RS=AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +	   ( AC(I,3) + ASI(I,J,K)*(BB(K)+BH(K)) )*H(I,J,K) + 
     +	    AC(I,4)*H(I,J+1,K) + AC(I,5)*H(I+1,J,K) + 
     +	    ASI(I,J,K)*(BL(K)*H(I,J,K-1) - THA(I,J,2)/DPI2(K))
     +	    - RH(I,J,K)
	  ZM=H(I,J,K)
	  H(I,J,K)=ZM - OMEGH*RS/(AC(I,3) + ASI(I,J,K)*(BB(K)+BH(K)))

	 ELSE
	  RS=AC(I,1)*H(I-1,J,K) + AC(I,2)*H(I,J-1,K) +
     +	   ( AC(I,3) + ASI(I,J,K)*BB(K) )*H(I,J,K) + 
     +	    AC(I,4)*H(I,J+1,K) + AC(I,5)*H(I+1,J,K) + 
     +	    ASI(I,J,K)*( BH(K)*H(I,J,K+1) + BL(K)*H(I,J,K-1) ) - 
     +	    RH(I,J,K)
	  ZM=H(I,J,K)
	  H(I,J,K)=ZM - OMEGH*RS/(AC(I,3) + ASI(I,J,K)*BB(K))
	 END IF	

	 DH(I,J,K)=H(I,J,K) - ZM 
         ZMRS=ZMRS + ABS(DH(I,J,K))
 	 IF (ABS(DH(I,J,K)).GT.THRS) THEN
	  IT=.FALSE.
	 END IF
 252    CONTINUE
 251    CONTINUE
 250    CONTINUE

	IF (AMOD(FLOAT(ITC),5.).EQ.0) THEN
         DHMAX=THRS/10.
         ZMRS=ZMRS/GPTS
         DO 324 K=2,NL-1
	  DO 325 J=2,NX-1
	  DO 326 I=2,NY-1
	   IF (ABS(DH(I,J,K)).GT.DHMAX) THEN
	    DHMAX=ABS(DH(I,J,K))
            IHM=I
            JHM=J
            KHM=K
	   END IF
 326      CONTINUE
 325      CONTINUE
 324     CONTINUE
	 WRITE(6,716)ZMRS,DHMAX,IHM,JHM,KHM,qe(ihm,jhm,khm)
 716     format(2e9.2,3i5,f8.3)
	END IF
	ZMRS=0.

	ITC=ITC+1
	IF (IT) THEN
	 PRINT*,'PHI CONVERGED.'
	 ITCT=ITCT + ITC
	 WRITE(6,*)ITC,ITCT,ITC1
 	 DO 260 J=1,NX
	 DO 261 I=1,NY
	  H(I,J,1)=H(I,J,2) + THA(I,J,1)*(PE(2)-PE(1))
          S(I,J,1)=S(I,J,2) + (THA(I,J,1)-THSB(1))*(PE(2)-PE(1))
	  H(I,J,NL)=H(I,J,NL-1) - THA(I,J,2)*(PE(NL)-PE(NL-1))
          S(I,J,NL)=S(I,J,NL-1) - (THA(I,J,2)-THSB(NL-1))*
     +         (PE(NL)-PE(NL-1))
 261    CONTINUE
 260    CONTINUE
	 IF (IITOT.GT.0) THEN
	 DO 263 K=1,NL
  	  DO 264 J=2,NX-1
	  DO 265 I=2,NY-1
	   H(I,J,K)=PART*H(I,J,K) + (1.-PART)*OLD(I,J,K)
 265      CONTINUE
 264      CONTINUE
 263     CONTINUE
	 END IF
         IF ( (itc.gt.itcold+10).and.(iitot.gt.30) ) then
          print*,'started diverging'
          go to 901 
         end if
         ITCOLD=ITC 
	 IF ((ITC.EQ.1).AND.(ITC1.EQ.NL-2)) THEN
	  PRINT*,'TOTAL CONVERGENCE.'
	 ELSE
	  IITOT=IITOT + 1
	  WRITE(6,22)IITOT
 22	  FORMAT(I4,' TOTAL ITERATION(S).')
	  IF (IITOT.GT.MAXXT) THEN
	   PRINT*,'TOO MANY TOTAL ITERATIONS.'
	   GO TO 901
	  ELSE
	   GO TO 900
	  END IF
	 END IF
	ELSE 
	 IF (ITC.LT.MAXX) THEN
	  GO TO 701
	 ELSE
	  PRINT*,'TOO MANY ITERATIONS FOR HGHT.'
	  ICON=.FALSE.
	  GO TO 901
	 END IF
	END IF

C*******************************************************

 901 	RETURN
	END	








