	PROGRAM QINVERTP
C************************************************
C****** Perturbation PV inversion program *******
C************************************************
C       Most of the variables used here have meanings similar to
C       those in the total PV inversion program. At the point the variables
C       are passed to the subroutine, those endine in 'B' are mean
C       quantities (QB for PV , MB for geopotential, SB for stream
C       function). Those ending in 'P' are perturbation quantities.
C       However, these variables are changed from the way they are read
C       in, according to some of the options discussed below.
C       There are 4 input files, q and h for the mean and q and h
C       for either the total fields. Perturbation fields are computed
C       from these.
C******* OPTIONS *************************
C
C        OMEGAS and OMEGAH are overrelaxation parameters for
C        S and H in subroutine BALP. PRT is the underrelaxation
C        parameter. These can all have the same values as in tthe
C        total inversion program. THRSH is the dimensional convergence
C        threshold (meters) and TSCAL and QSCAL are scale factors
C        to multiply the input boundary theta and PV respectively
C        (generally = 1.0).
C        
C        INLIN = option for nonlinear terms. If INLIN = 1, there are
C        no terms dropped in the perturbation inversion equations. The
C        equations are still linear, but the nonlinear terms are
C        hidden in the coefficents of the differential operator. The
C        "mean variables" are redefined to include the actual mean
C        field plus 1/2 the perturbation field. If INLIN=0, nonlinear
C        terms are dropped altogether. Usually, INLIN =1, otherwise,
C        the sum of piecewise solutions will not equal the total 
C        perturbation.
C
C        IQD = option to make value of q' conditional on some other
C        field. Suppose you wanted the PV perturbation to be only
C        the perturbation PV in saturated air, you could read in 
C        a relative humidity file here and with CRIT, specify the
C        threshold value of relative humidity below which q'=0. There
C        is also a hardwired option to allow only positive q' in saturated
C        air: this can be changed.
C
C      SUB BALP
C
C       These options should be pretty self-explanatory, beginning
C       with the number and list of the output perturbation h and s
C       levels, the number of inversions to be done and then
C       the number and list of pert. PV levels to be included.
C       1=lower boundary theta
C       2 to nl-1 are interior pert. PV levels, bottome to top
C       NL=upper boundary perturbation theta.
C
C       If you don't choose either boundary theta field, homogeneous
C       Neumann conditions are applied at the top and bottom on both
C       'h' and 's'.
C
C       IBC = option for lateral boundary conditions.
C        
C       IBC=0 => homogeneous Diriclet conditions.
C       IBC=1 => 's' and 'h' at the boundaries are equal to the
C                 full perturbation
C       IBC=2 => you will read in a file with an estimate of the
C                 interior and boundary values. This option
C                 is useful for "nesting" the inversion, i.e., using
C                 the boundary conditions from a calculation
C                 done on a larger grid.
C**************************************************************
C****************************************************************
	PARAMETER (NL=10)
	PARAMETER (NY=21)
	PARAMETER (NX=45)
	REAL QP(NY,NX,NL),QB(NY,NX,NL),AP(NY),APP(NY),APM(NY),
     +       MP(NY,NX,NL),MB(NY,NX,NL),THB(NY,NX,2),ZHDR(8),
     +       SIP(NY,NX,NL),SB(NY,NX,NL),THP(NY,NX,2),HND,FM(NY,NX),
     +	     FC(NY,NX),SIGM,AA,BETA,PI(NL),PR(NL),PIF(NL),
     +	     KAP,R,CP,CV,FF,LL,PII,MI,FRC,A(NY,5),PRT,QMIN,
     +	     MF(NY,NX),LATT(NY,NX),DPF(NY,NX,NL),QPO(NY,NX,NL)
	INTEGER HDR(8),BRES,IQP(NL)
	CHARACTER*50 FNM(10),DPFIL
C        DATA PR/ 1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55,
C     +           0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15/
	DATA PR/ 1.0, 0.85, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1/
C        DATA PR/ 1.000000, .8761213, .7636118, .6618214, .5701128,
C     +    .4878614, .4144568, .3493018, .2918134, .2414231,
C     +    .1975774, .1597475, .1273806, .1000000/
        QMIN=0.01
        MAX=250
        MAXT=100
	PRINT*,'OMEGAS, OMEGAH, PRT, THRSH, TSCAL, QSCAL ?'
        READ(5,*)OMEGAS
        READ(5,*)OMEGAH
        READ(5,*)PRT
        READ(5,*)THRSH
        READ(5,*)TSCAL
        READ(5,*)QSCAL
	PRINT*,'ENTER FILE NAMES FOR MEANS (Q,H), TOTAL FIELDS (Q,H),
     + AND OUTPUT (B12) FILES.'
        READ(5,*)FNM(1)
        READ(5,*)FNM(2)
        READ(5,*)FNM(3)
        READ(5,*)FNM(4)
        READ(5,*)FNM(5)
        PRINT*,'ENTER 1 FOR SPHERICAL, 2 FOR XY PROJ.'
        READ(5,*)IMAP

	DO 200 KL=1,4
	 OPEN(KL,FILE=FNM(KL),STATUS='OLD')
	 READ(KL,*)(ZHDR(I),I=1,8)
 200    CONTINUE
	 OPEN(14,FILE=FNM(5),STATUS='new')
	 WRITE(14,*)(ZHDR(I),I=1,8)
         DO 199 I=1,8
          HDR(I)=NINT(ZHDR(I))
 199     CONTINUE

	PII=4.*ATAN(1.)
	AA=2.E7/PII
	DPI=50.
	CP=1004.5
	R=2.*CP/7.
	CV=CP-R
	FF=1.E-4
	SIGM=ZHDR(5)/ZHDR(6)	
	KAP=R/CP
	PO=1.E5
	MI=9999.90
	GG=9.81
        IF (IMAP.EQ.1) THEN
	 LL=AA*ZHDR(5)*PII/180.
         DO 101 I=1,HDR(8)
         AP(I)=COS( PII*(ZHDR(3) - (I-1)*ZHDR(6))/180. )
	 APM(I)=COS( PII*(ZHDR(3) - (I-1.5)*ZHDR(6))/180. )
	 APP(I)=COS( PII*(ZHDR(3) - (I-0.5)*ZHDR(6))/180. )
  	 A(I,1)=SIGM*SIGM*APM(I)/AP(I)
	 A(I,2)=1./( AP(I)*AP(I) )
	 A(I,3)=-( 2. + SIGM*SIGM*AP(I)*(APM(I)+APP(I)) )/
     +	    ( AP(I)*AP(I) )
	 A(I,4)=1./( AP(I)*AP(I) )
	 A(I,5)=SIGM*SIGM*APP(I)/AP(I)
         DO 104 J=1,HDR(7)
          FC(I,J)=1.458*SIN( PII*(ZHDR(3) - (I-1)*ZHDR(6))/180. )
          MF(I,J)=1.
          FM(I,J)=FC(I,J)/MF(I,J)
 104     CONTINUE
 101     CONTINUE
        ELSE
         LL=ZHDR(5)*1.E3
         OPEN(23,FILE='latlon',STATUS='OLD')
         READ(23,*)(ZHDR(I),I=1,8)
         DO 105 I=1,HDR(8)
          READ(23,*)(MF(I,J),J=1,HDR(7))
 105     CONTINUE        
         DO 103 I=1,HDR(8)
          READ(23,*,END=103)(LATT(I,J),J=1,HDR(7))
 103     CONTINUE        
         DO 102 I=1,HDR(8)
         AP(I)=1.
  	 A(I,1)=SIGM*SIGM
	 A(I,2)=1.
	 A(I,3)=-2*( 1. + SIGM*SIGM )
	 A(I,4)=1.
	 A(I,5)=SIGM*SIGM
         DO 106 J=1,HDR(7)
          FC(I,J)=1.458*SIN( PII*LATT(I,J)/180. )
          MF(I,J)=MF(I,J)*MF(I,J)
          FM(I,J)=FC(I,J)/MF(I,J)
 106     CONTINUE
 102     CONTINUE
        END IF
        THO=FF*FF*LL*LL/DPI
	FRC=DPI*THO/(FF*FF*LL*LL)
	QCONST=1.E6*KAP*GG*CP*FF*THO/(DPI*PO)
	BETA=1.458*LL/AA
	HND=DPI*THO/GG
        THRSH=THRSH/HND
	WRITE(6,*)FRC,QCONST,THO,HND,THRSH

	DO 204 K=1,NL
	 PI(K)=CP*( PR(K)**KAP )/DPI
	 PIF(K)=(DPI*PI(K)/CP)**2.5
 204    CONTINUE
C*********************************************************************
        DO 230 K=1,2
         DO 231 I=1,HDR(8)
         READ(1,*)(THB(I,J,K),J=1,HDR(7))
 231     CONTINUE
 230    CONTINUE
	DO 232 K=2,NL-1
         DO 233 I=1,HDR(8)
          READ(1,*)(QB(I,J,K),J=1,HDR(7))
 233     CONTINUE
 232    CONTINUE
C********* Read in hght and stream fcn fields *******************************
	DO 240 K=1,NL
	 DO 241 I=1,HDR(8)
	  READ(2,*,END=238)(MB(I,J,K),J=1,HDR(7))
 238 	  READ(4,*,END=239)(MP(I,J,K),J=1,HDR(7))
 241     CONTINUE
 240    CONTINUE
 239	DO 242 K=1,NL
         DO 243 I=1,HDR(8)
	  READ(2,*,END=11)(SB(I,J,K),J=1,HDR(7))
 11       READ(4,*,END=12)(SIP(I,J,K),J=1,HDR(7))
 243     CONTINUE
 242    CONTINUE
C**************** Read in theta and PV data *********************************
 12 	DO 250 K=1,2
	 DO 251 I=1,HDR(8)
	  READ(3,*)(THP(I,J,K),J=1,HDR(7))
	  DO 252 J=1,HDR(7)
	   THB(I,J,K)=THB(I,J,K)/THO
	   THP(I,J,K)=TSCAL*THP(I,J,K)/THO - THB(I,J,K)
 252      CONTINUE
 251     CONTINUE
 250    CONTINUE
 	DO 260 K=2,NL-1
         DO 261 I=1,HDR(8)
	  READ(3,*,END=31)(QP(I,J,K),J=1,HDR(7))
 261     CONTINUE
 260    CONTINUE

C********* Nondimensionalize variables ******************************
 31       PRINT*,'INCLUDE NONLINEAR TERMS? (1=Y).'
        READ(5,*)INLIN
        IF (INLIN.EQ.1) THEN 
         CNLIN=0.5
        ELSE
         CNLIN=0.
        END IF
 	DO 270 K=1,NL
         DO 271 J=1,HDR(7)
	  DO 272 I=1,HDR(8)
	   MP(I,J,K)=MP(I,J,K)/HND
	   SIP(I,J,K)=SIP(I,J,K)/HND
	   MB(I,J,K)=MB(I,J,K)/HND
	   SB(I,J,K)=SB(I,J,K)/HND
	   MP(I,J,K)=MP(I,J,K)-MB(I,J,K)
	   SIP(I,J,K)=SIP(I,J,K)-SB(I,J,K)
	   MB(I,J,K)=MB(I,J,K) + CNLIN*MP(I,J,K)
	   SB(I,J,K)=SB(I,J,K) + CNLIN*SIP(I,J,K)
 272      CONTINUE
 271     CONTINUE
 270    CONTINUE
 	DO 280 K=2,NL-1
	 DO 281 J=1,HDR(7)
	  DO 282 I=1,HDR(8)
	   IF (QP(I,J,K).EQ.MI) THEN
	    QP(I,J,K)=0.
	    GO TO 282
	   END IF
	   IF (QP(I,J,K).LT.1.E2*QMIN) THEN
	    QP(I,J,K)=PIF(K)*QMIN/QCONST
	   ELSE
	    QP(I,J,K)=PIF(K)*QP(I,J,K)/(1.E2*QCONST)
	   END IF
	   IF (QB(I,J,K).LT.1.E2*QMIN) THEN
	    QB(I,J,K)=PIF(K)*QMIN/QCONST
	   ELSE
	    QB(I,J,K)=PIF(K)*QB(I,J,K)/(1.E2*QCONST)
	   END IF
	   QP(I,J,K)=(QP(I,J,K)-QB(I,J,K))/MF(I,J)
 282      CONTINUE
 281     CONTINUE
 280    CONTINUE
        PRINT*,'ENTER "1" IF QPRIME.NE.0 DEPENDS ON ANOTHER FIELD.'
        READ*,IQD
        IF (IQD.EQ.1) THEN
         PRINT*,'ENTER THE FILE, THE DAY AND NUM LEVELS PER DAY.'
         READ(5,*)DPFIL
         READ(5,*)NDAYS
         READ(5,*)LPD
         OPEN(18,FILE=DPFIL,STATUS='OLD')
         READ(18,*)(ZHDR(I),I=1,8)
         DO 290 NN=1,NDAYS
          DO 291 K=1,LPD
           DO 292 I=1,HDR(8)
            READ(18,*,END=297)(DPF(I,J,K),J=1,HDR(7))
 292       CONTINUE
 291      CONTINUE
 290     CONTINUE
         OPEN(19,FILE='DT:[CDAVIS]QPR.DAT',STATUS='NEW')
 297     WRITE(19,*)(ZHDR(I),I=1,8)
         DO 300 K=2,NL-1
          PRINT*,'ENTER VALUE OF THIS FIELD BELOW WHICH QPRM=0.'
          READ*,CRIT
          IQP(K)=0
          DO 301 J=2,HDR(7)-1
          DO 302 I=2,HDR(8)-1          
           IF ((DPF(I,J,K).LT.CRIT).OR.(QP(I,J,K).LT.0.)) THEN
            QP(I,J,K)=0.
           ELSE
            IQP(K)=IQP(K) + 1
           END IF
 302      CONTINUE
 301      CONTINUE
          IF (IQP(K).GT.0) THEN
           WRITE(6,*)IQP(K),K
           DO 310 I=1,HDR(8)
            DO 312 J=1,HDR(7)
             QPO(I,J,K)=QP(I,J,K)*1.E2*QCONST/PIF(K)
 312        CONTINUE
            WRITE(19,399)(QPO(I,J,K),J=1,HDR(7))
 310       CONTINUE
          END IF
 300     CONTINUE
 399     FORMAT(10F8.2)
        END IF    
C*********** Routine to solve for balanced pert flow by S.O.R. **************
	CALL BALP(MP,MB,SIP,SB,THP,QP,FC,FM,MF,AP,A,PI,HND,
     +	  OMEGAS,OMEGAH,PRT,THRSH,BETA,FRC,MAX,MAXT,SIGM)
	STOP
	END
C**************************************************************************
	SUBROUTINE BALP(H,HB,S,SBR,TPR,Q,FCO,FCM,MFC,APS,AC,PE,
     +	  HNDM,OMEGS,OMEGH,PART,THRS,BET,FR,MAXX,MAXXT,SIG)
	PARAMETER (NL=10)
	PARAMETER (NY=21)
	PARAMETER (NX=45)
	REAL  H(NY,NX,NL),HB(NY,NX,NL),HP(NY,NX,NL),MI,PART,
     +	   SBR(NY,NX,NL),S(NY,NX,NL),SP(NY,NX,NL),Q(NY,NX,NL),SIG,
     +     ZM,AC(NY,5),SLL(NY,NX,NL),SPP(NY,NX,NL),SLP(NY,NX,NL),
     +	   STB(NY,NX,NL),AVO(NY,NX,NL),RHS(NY,NX,NL),QP(NY,NX,NL),
     +	   ZNC(NY,NX),BET,FR,FCO(NY,NX),TPR(NY,NX,2),
     +	   APS(NY),RS,ASI(NY,NX,NL),BSI(NY,NX,NL),APHI(NY,NX,NL),
     +	   HRHS(NY,NX,NL),SRHS(NY,NX,NL),TP(NY,NX,2),OS(NY,NX,NL),
     +	   OH(NY,NX,NL),BB(NL),PE(NL),BH(NL),BL(NL),DPI2(NL),HNDM,
     +     SISUM(NY,NX,NL),HTSUM(NY,NX,NL),FCM(NY,NX),MFC(NY,NX),
     +     XHDR(8)
	INTEGER QLV(NL),SIOUT(NL),HOUT(NL),GPTS
        CHARACTER*30 IGFIL
	LOGICAL IT,ICON
	MI=9999.90
        WRITE(6,*)THRS,OMEGS,OMEGH,PART
	GPTS=NY*NX*NL
        write(6,600)FR,SIG
 600    FORMAT(' FR=',F10.3,' AND SIG=',F10.3)
	WRITE(6,601)GPTS,NY,NX,NL
 601    FORMAT(I6,' gridpoints in domain;',I4,' X',I4,' X',I4)
C********** Set coefficients ********************************
	DO 200 I=2,NY-1
         DO 201 J=2,NX-1
	  ZNC(I,J)=2.*FR*SIG*SIG*MFC(I,J)/(APS(I)*APS(I))
 201     CONTINUE
 200    CONTINUE
	DO 202 K=2,NL-1
	 BB(K)=-2./( (PE(K+1)-PE(K))*(PE(K)-PE(K-1)) )
	 BH(K)=2./( (PE(K+1)-PE(K))*(PE(K+1)-PE(K-1)) )
	 BL(K)=2./( (PE(K+1)-PE(K-1))*(PE(K)-PE(K-1)) )
	 DPI2(K)=(PE(K+1)-PE(K-1))/2.
         ivneg = 0
	 DO 203 J=2,NX-1
	 DO 204 I=2,NY-1
	  SLL(I,J,K)=ZNC(I,J)*(SBR(I,J+1,K)+SBR(I,J-1,K)-2.*SBR(I,J,K))
	  SPP(I,J,K)=ZNC(I,J)*(SBR(I-1,J,K)+SBR(I+1,J,K)-2.*SBR(I,J,K))
	  AVO(I,J,K)=FCM(I,J) + FR*( AC(I,1)*SBR(I-1,J,K) +
     +	     AC(I,2)*SBR(I,J-1,K) + AC(I,3)*SBR(I,J,K) + 
     +	     AC(I,4)*SBR(I,J+1,K) + AC(I,5)*SBR(I+1,J,K) )
	  SLP(I,J,K)=ZNC(I,J)*(SBR(I-1,J+1,K)-SBR(I-1,J-1,K)-
     +	     SBR(I+1,J+1,K)+SBR(I+1,J-1,K))/2.     !COEFF IS REALLY  2./4.
          STB(I,J,K)=BH(K)*HB(I,J,K+1) + BL(K)*HB(I,J,K-1) +
     +       BB(K)*HB(I,J,K)
       	  IF ( AVO(I,J,K).LT.0.001 ) THEN
C	   WRITE(6,96)AVO(I,J,K),I,J,K
	   AVO(I,J,K)=0.01
           ivneg=ivneg + 1
 96	   FORMAT(' negative vorticity',f10.3,' at i j k=',3I3)
	  END IF
	  IF (STB(I,J,K).LT.0.001) THEN
c	   WRITE(6,95)stb(i,j,k),I,J,K
           stb(i,j,k)=0.01
 95	   FORMAT(' negative stability',f10.3,' at i j k=',3I3)
	  END IF
 204     CONTINUE
 203     CONTINUE
         write(6,*)ivneg,k
C**************************************************************
	 DO 206 J=2,NX-1
	 DO 207 I=2,NY-1
	  ASI(I,J,K)=BB(K)*AVO(I,J,K)/(FR*STB(I,J,K)*AC(I,3))
	  BSI(I,J,K)=1. + ASI(I,J,K)*FCO(I,J)
C
C******** ELLIPTICITY CHECK *************************
C
C          ZRAD=(ASI(I,J,K)*SLP(I,J,K)/2.)**2 - (1. + ASI(I,J,K)*
C     +      (FCO(I,J)+SLL(I,J,K)))*(1. + ASI(I,J,K)*
C     +      (FCO(I,J)+SPP(I,J,K)))             
C          IF (ZRAD.GE.0.) THEN
C           WRITE(6,*)I,J,K,ZRAD,SLP(I,J,K)
C           IF (SLL(I,J,K) + FCO(I,J).LT.0.) THEN
C            WRITE(6,*)SLL(I,J,K),FCO(I,J)
C            SLL(I,J,K)=-FCO(I,J)
C           END IF
C           IF (SPP(I,J,K) + FCO(I,J).LT.0.) THEN
C            WRITE(6,*)FCO(I,J),SPP(I,J,K)
C            SPP(I,J,K)=-FCO(I,J)
C           END IF
C           ZRAD=(ASI(I,J,K)*SLP(I,J,K)/2.)**2 - (1. + ASI(I,J,K)*
C     +      (FCO(I,J)+SLL(I,J,K)))*(1. + ASI(I,J,K)*
C     +      (FCO(I,J)+SPP(I,J,K)))             
C           IF (ZRAD.GE.0.) THEN
C            WRITE(6,*)ZRAD
C            SLP(I,J,K)=0.
C             (2.*SIGN(1.,SLP(I,J,K))/ASI(I,J,K))*
c     +       SQRT(-0.001 + (1. + ASI(I,J,K)*(FCO(I,J)+SLL(I,J,K)))*
c     +            (1. + ASI(I,J,K)*(FCO(I,J)+SPP(I,J,K))) )            
c            WRITE(6,*)SLP(I,J,K)
C           END IF
C          END IF
C*******************************************
	  BI=FCO(I,J)*AC(I,3) - 2.*(SLL(I,J,K) + SPP(I,J,K))	 
          IF (BI.GT.0.) THEN
C           PRINT*,'BI GT 0.'
C           WRITE(6,*)I,J,BI
           BI=0.
          END IF
	  APHI(I,J,K)=BI/(FR*STB(I,J,K)*AC(I,3))
 207     CONTINUE
 206     CONTINUE
 202     CONTINUE
C***********************************************************
	PRINT*,'ENTER # AND LIST OF OUTPUT HT LEVELS'
	READ*,NHO,(HOUT(I),I=1,NHO)
	PRINT*,'ENTER # AND LIST OF OUTPUT PSIT LEVELS'
	READ*,NSIO,(SIOUT(I),I=1,NSIO)
C******** Determine the number of fields to solve for *******
	PRINT*,'ENTER THE NUMBER OF INVERSIONS TO BE DONE.'
	READ*,NOUT
        write(6,610)hout(nho)
 610    format(' highest output phi level, number is',i4)
        write(6,611)siout(nsio)
 611    format(' highest output psi level, number is',i4)
        write(6,612)nout
 612    format(' number of inversions=',i4)

	DO 210 IH=1,NOUT		!Begin total iteration loop
	 PRINT*,'ENTER # AND LIST OF PV LEVELS.'
	 READ*,NMLV,(QLV(I),I=1,NMLV)
         write(6,620)nmlv
 620     format(' number of levels=',i4)
C******* Initialize fields, set boundary conditions *******
	 DO 220 J=1,NX
	 DO 221 I=1,NY
	  TP(I,J,1)=0.
	  TP(I,J,2)=0.
	  DO 222 K=2,NL-1
	   QP(I,J,K)=0.
 222      CONTINUE
 221     CONTINUE
 220     CONTINUE

	 IF (NMLV.EQ.NL) THEN	!USE ALL LEVELS
	  DO 230 J=1,NX
	  DO 231 I=1,NY
	   TP(I,J,1)=TPR(I,J,1)
	   TP(I,J,2)=TPR(I,J,2)
	   DO 232 K=2,NL-1
	    QP(I,J,K)=Q(I,J,K)
 232      CONTINUE
 231     CONTINUE
 230     CONTINUE
	  GO TO 114
	 END IF
C************************************
	 DO 240 KL=1,NMLV
	  IF (QLV(KL).EQ.1) THEN
	   WRITE(6,602)kl,qlv(kl)
 602       FORMAT(' level ',I4,' field',i4)
	   DO 241 J=1,NX
	   DO 242 I=1,NY
	    TP(I,J,1)=TPR(I,J,1)
 242       CONTINUE
 241       CONTINUE
	  ELSE IF (QLV(KL).EQ.NL) THEN
	   WRITE(6,602)kl,qlv(kl)
	   DO 243 J=1,NX
	   DO 244 I=1,NY
	    TP(I,J,2)=TPR(I,J,2)
 244       CONTINUE
 243       CONTINUE
	  END IF
	  DO 245 K=2,NL-1
	   IF (K.EQ.QLV(KL)) THEN
       	    WRITE(6,602)kl,qlv(kl)
	    DO 246 J=1,NX
	    DO 247 I=1,NY
	     QP(I,J,K)=Q(I,J,K)
 247        CONTINUE
 246        CONTINUE
	   END IF
 245      CONTINUE
 240     CONTINUE
 114     PRINT*,'ENTER "1" FOR TOTAL PERT B.C., "0" FOR HOMOGEN B.C.'
         print*,'ENTER "2" FOR INIT GUESS FILE.'
         READ(5,*)IBC
         IF (IBC.EQ.2) THEN
          PRINT*,'ENTER FILENAME.'
          READ(5,*)IGFIL
          OPEN(24,FILE=IGFIL,STATUS='OLD')
          READ(24,*)(XHDR(I),I=1,8)
	  DO 250 K=1,NL
	   DO 251 I=1,NY
            READ(24,*)(HP(I,J,K),J=1,NX)
            DO 2511 J=1,NX
             HP(I,J,K)=HP(I,J,K)/HNDM
 2511       CONTINUE
 251       CONTINUE
 250      CONTINUE
	  DO 252 K=1,NL
	   DO 253 I=1,NY
            READ(24,*,END=253)(SP(I,J,K),J=1,NX)
            DO 2531 J=1,NX
             SP(I,J,K)=SP(I,J,K)/HNDM
 2531       CONTINUE
 253       CONTINUE
 252      CONTINUE
         ELSE IF (IBC.EQ.1) THEN
	  DO 254 K=1,NL
	   DO 255 J=1,NX
	   DO 256 I=1,NY
	    HP(I,J,K)=H(I,J,K) - HTSUM(I,J,K)	!INITIALIZE H1
	    SP(I,J,K)=S(I,J,K) - SISUM(I,J,K)	!INITIALIZE S1
 256       CONTINUE
 255       CONTINUE
 254      CONTINUE
	 ELSE
	  DO 257 K=1,NL
	   DO 258 J=1,NX
	   DO 259 I=1,NY
	    HP(I,J,K)=0.	!INITIALIZE H1
	    SP(I,J,K)=0. 	!INITIALIZE S1
 259       CONTINUE
 258       CONTINUE
 257      CONTINUE
	 END IF
C********* Calculate upper and lower initial boundary values ***********
	 DO 260 J=1,NX
	 DO 261 I=1,NY
	  HP(I,J,1)=HP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
	  SP(I,J,1)=SP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
	  HP(I,J,NL)=HP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
	  SP(I,J,NL)=SP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
 261     CONTINUE
 260     CONTINUE
C********************************************************
C*********** Begin iterations ***************************
	 IITOT=0
	 ITC=0
 900	CONTINUE		!Total iteration

C********** Calculate the RHS of the psi equation *********
	DO 270 K=1,NL
	 DO 271 J=1,NX
	 DO 272 I=1,NY
	  OS(I,J,K)=SP(I,J,K)
	  OH(I,J,K)=HP(I,J,K)
 272     CONTINUE
 271    CONTINUE
 270    CONTINUE

	DO 280 K=2,NL-1
	DO 281 J=2,NX-1
	 DO 282 I=2,NY-1
          R1BS=( SBR(I,J+1,K+1)-SBR(I,J-1,K+1)-SBR(I,J+1,K-1)+
     +	    SBR(I,J-1,K-1) )/(4.*DPI2(K))
	  R1BH=( HB(I,J+1,K+1)-HB(I,J-1,K+1)-HB(I,J+1,K-1)+
     +      HB(I,J-1,K-1) )/(4.*DPI2(K))
	  R2BS=( SBR(I-1,J,K+1)-SBR(I+1,J,K+1)-SBR(I-1,J,K-1)+
     +	    SBR(I+1,J,K-1) )/(4.*DPI2(K))
          R2BH=( HB(I-1,J,K+1)-HB(I+1,J,K+1)-HB(I-1,J,K-1)+
     +      HB(I+1,J,K-1) )/(4.*DPI2(K))
	  R1PS=( SP(I,J+1,K+1)-SP(I,J-1,K+1)-SP(I,J+1,K-1)+
     +	    SP(I,J-1,K-1) )/(4.*DPI2(K))
	  R1PH=( HP(I,J+1,K+1)-HP(I,J-1,K+1)-HP(I,J+1,K-1)+
     +	    HP(I,J-1,K-1) )/(4.*DPI2(K))
	  R2PS=( SP(I-1,J,K+1)-SP(I+1,J,K+1)-SP(I-1,J,K-1)+
     +	    SP(I+1,J,K-1) )/(4.*DPI2(K))
	  R2PH=( HP(I-1,J,K+1)-HP(I+1,J,K+1)-HP(I-1,J,K-1)+
     +	    HP(I+1,J,K-1) )/(4.*DPI2(K))
	  RHS(I,J,K)=QP(I,J,K) + FR*( (R1BS*R1PH + R1BH*R1PS)/
     +	    (APS(I)*APS(I)) + SIG*SIG*(R2BS*R2PH + R2BH*R2PS) )
          SRHS(I,J,K)=( RHS(I,J,K) - AVO(I,J,K)*(BH(K)*HP(I,J,K+1)+ 
     +	    BL(K)*HP(I,J,K-1)) )/(FR*STB(I,J,K)) + ASI(I,J,K)*
     +	    ( AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) + AC(I,4)*
     +	     HP(I,J+1,K) + AC(I,5)*HP(I+1,J,K) )
 282     CONTINUE
 281    CONTINUE
 280    CONTINUE

C************* Iteration for psi (2-D) **********************
        ITCC=0
	DO 290 K=2,NL-1
	ITC=0
 800 	IT=.TRUE.
         ZMRS=0
	 DO 291 J=2,NX-1
	 DO 292 I=2,NY-1
	  RSA=AC(I,1)*SP(I-1,J,K) + AC(I,2)*
     +	   SP(I,J-1,K) + AC(I,3)*SP(I,J,K) + 
     +	   AC(I,4)*SP(I,J+1,K) + AC(I,5)*SP(I+1,J,K) 
	  SXX=SP(I,J+1,K)+SP(I,J-1,K) -2.*SP(I,J,K)
	  SYY=SP(I-1,J,K)+SP(I+1,J,K)-2.*SP(I,J,K)
	  SXY=( SP(I-1,J+1,K)-SP(I+1,J+1,K)-SP(I-1,J-1,K)+
     +	    SP(I+1,J-1,K) )/4.
          BETAS=SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +     (SP(I-1,J,K)-SP(I+1,J,K))/4. + (FCO(I,J+1)-FCO(I,J-1))*
     +     (SP(I,J+1,K)-SP(I,J-1,K))/4.
	  RS=BSI(I,J,K)*RSA + ASI(I,J,K)*( BETAS + SLL(I,J,K)*SYY +
     +	     SPP(I,J,K)*SXX - SLP(I,J,K)*SXY ) - SRHS(I,J,K)

	  ZSI=SP(I,J,K)
	  SP(I,J,K)=ZSI - OMEGS*RS/( BSI(I,J,K)*AC(I,3) -
     +	    2.*ASI(I,J,K)*(SLL(I,J,K) + SPP(I,J,K)) )
          ZMRS=ZMRS + ABS(ZSI-SP(I,J,K))/GPTS

 	  IF (ABS(SP(I,J,K)-ZSI).GT.THRS) THEN
	   IT=.FALSE.
	  END IF
 292     CONTINUE
 291     CONTINUE

	ITC=ITC+1
	IF (AMOD(FLOAT(ITC),10.).EQ.0) THEN
	 WRITE(6,*)ITC,ZMRS
	END IF

	IF (IT) THEN
	 ICON=.TRUE.
         WRITE(6,603)ITC,K
         IF (ITC.GT.1) ITCC=1
 603     FORMAT(I4,' ITERATIONS AT LEVEL',I4)
	ELSE 
	 IF (ITC.LT.MAXX) THEN
	  GO TO 800
	 ELSE
	  PRINT*,'TOO MANY ITERATIONS FOR PSI.'
	  ICON=.FALSE.
	  GO TO 901
	 END IF
	END IF

 290    CONTINUE

	PRINT*,'PSI CONVERGED.'
	IF (IITOT.GT.0) THEN
	DO 300 K=1,NL
	 DO 301 J=2,NX-1
	 DO 302 I=2,NY-1
	  SP(I,J,K)=PART*SP(I,J,K) + (1.-PART)*OS(I,J,K)
 302     CONTINUE
 301     CONTINUE
 300    CONTINUE
	END IF

C**************************************************************
C********* Calculate the RHS of the PV+BALANCE equation *******

 700	DO 310 K=2,NL-1
	 DO 311 J=2,NX-1
	 DO 312 I=2,NY-1
	  RH1=(2./AC(I,3))*(SLL(I,J,K) + SPP(I,J,K))* 
     +	     ( AC(I,1)*SP(I-1,J,K) + AC(I,2)*SP(I,J-1,K) + 
     +	     AC(I,4)*SP(I,J+1,K) + AC(I,5)*SP(I+1,J,K) )
          BETAS=SIG*SIG*(FCO(I-1,J)-FCO(I+1,J))*
     +     (SP(I-1,J,K)-SP(I+1,J,K))/4. + (FCO(I,J+1)-FCO(I,J-1))*
     +     (SP(I,J+1,K)-SP(I,J-1,K))/4.
     	  RH2=BETAS + SLL(I,J,K)*(SP(I-1,J,K)+SP(I+1,J,K)) +
     +	    SPP(I,J,K)*(SP(I,J-1,K)+SP(I,J+1,K)) -
     +	    SLP(I,J,K)*(SP(I-1,J+1,K)-SP(I-1,J-1,K)-
     +	     SP(I+1,J+1,K)+SP(I+1,J-1,K))/4.
	  HRHS(I,J,K)=APHI(I,J,K)*RHS(I,J,K) + RH1 + RH2
 312     CONTINUE
 311     CONTINUE
 310    CONTINUE

C************* Solve for phi with 3-D SOR *****************
 	ITC=0
        zmrs=0.
 701 	IT=.TRUE.
	
	DO 320 K=2,NL-1
	DO 321 J=2,NX-1
	DO 322 I=2,NY-1
	 IF (K.EQ.2) THEN
	  RS=AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) +
     +	   ( AC(I,3) + APHI(I,J,K)*(BB(K)+BL(K))*AVO(I,J,K) )*
     +	    HP(I,J,K) + AC(I,4)*HP(I,J+1,K) + AC(I,5)*
     +	    HP(I+1,J,K) + APHI(I,J,K)*AVO(I,J,K)*( BH(K)*
     +	     HP(I,J,K+1) + TP(I,J,1)/DPI2(K) ) - HRHS(I,J,K)
	  ZM=HP(I,J,K)
	  HP(I,J,K)=ZM - OMEGH*RS/( AC(I,3) + APHI(I,J,K)*
     +	      (BB(K)+BL(K))*AVO(I,J,K) )

	 ELSE IF (K.EQ.NL-1) THEN
	  RS=AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) +
     +	   ( AC(I,3) + APHI(I,J,K)*(BB(K)+BH(K))*AVO(I,J,K) )*
     +	    HP(I,J,K) + AC(I,4)*HP(I,J+1,K) + AC(I,5)*
     +	    HP(I+1,J,K) + APHI(I,J,K)*AVO(I,J,K)*( BL(K)*
     +	     HP(I,J,K-1) - TP(I,J,2)/DPI2(K) ) - HRHS(I,J,K)
	  ZM=HP(I,J,K)
	  HP(I,J,K)=ZM - OMEGH*RS/( AC(I,3) + APHI(I,J,K)*
     +	      (BB(K)+BH(K))*AVO(I,J,K) )

	 ELSE
	  RS=AC(I,1)*HP(I-1,J,K) + AC(I,2)*HP(I,J-1,K) +
     +	   ( AC(I,3) + APHI(I,J,K)*BB(K)*AVO(I,J,K) )*
     +	    HP(I,J,K) + AC(I,4)*HP(I,J+1,K) + AC(I,5)*HP(I+1,J,K) + 
     +	    APHI(I,J,K)*AVO(I,J,K)*( BH(K)*HP(I,J,K+1) + 
     +	     BL(K)*HP(I,J,K-1) ) - HRHS(I,J,K)
	  ZM=HP(I,J,K)
	  HP(I,J,K)=ZM-OMEGH*RS/( AC(I,3) + 
     +	     BB(K)*APHI(I,J,K)*AVO(I,J,K) )
	 END IF	

         ZMRS=ZMRS + ABS(ZM-HP(I,J,K))/GPTS
 	 IF (ABS(ZM-HP(I,J,K)).GT.THRS/2.) THEN
	  IT=.FALSE.
	 END IF
 322    CONTINUE
 321    CONTINUE
 320    CONTINUE

	IF (AMOD(FLOAT(ITC),10.).EQ.0) THEN
	 WRITE(6,*)ITC,ZMRS
	END IF
	ZMRS=0.

	ITC=ITC+1
	IF (IT) THEN
	 PRINT*,'PHI CONVERGED.'
         WRITE(6,606)ITC
 606     FORMAT(' Phi converged in',I4,' iterations.')
	 DO 330 J=1,NX
	 DO 331 I=1,NY
	  HP(I,J,1)=HP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
	  SP(I,J,1)=SP(I,J,2) + TP(I,J,1)*(PE(2)-PE(1))
	  HP(I,J,NL)=HP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
	  SP(I,J,NL)=SP(I,J,NL-1) - TP(I,J,2)*(PE(NL)-PE(NL-1))
 331     CONTINUE
 330     CONTINUE
	 IF (IITOT.GT.0) THEN
	  DO 340 K=1,NL
	  DO 341 J=2,NX-1
	  DO 342 I=2,NY-1
	   HP(I,J,K)=PART*HP(I,J,K) + (1.-PART)*OH(I,J,K)
 342      CONTINUE
 341      CONTINUE
 340      CONTINUE
	 END IF
	 IF ((ITC.EQ.1).AND.(ITCC.EQ.0)) THEN
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

C********** Write out phi and psi fields ***********************
 901	M=1
	MH=1
	DO 350 K=1,NL
	 DO 351 J=1,NX
	 DO 352 I=1,NY
	  SISUM(I,J,K)=SISUM(I,J,K) + SP(I,J,K)
	  HTSUM(I,J,K)=HTSUM(I,J,K) + HP(I,J,K)
 352     CONTINUE
 351     CONTINUE
	 IF (HOUT(MH).EQ.K) THEN
	  WRITE(6,615)K,MH
 615      format(2i6)
	  DO 353 I=1,NY
	   DO 354 J=1,NX
	    HP(I,J,K)=HP(I,J,K)*HNDM
 354       CONTINUE
 353      CONTINUE
	  DO 355 I=1,NY
	   WRITE(14,991)(HP(I,J,K),J=1,NX)
 355      CONTINUE
	  DO 357 I=1,NY
	   DO 358 J=1,NX
	    HP(I,J,K)=HP(I,J,K)/HNDM
 358       CONTINUE
 357      CONTINUE
          MH=MH+1
	 END IF
 350    CONTINUE
	DO 360 K=1,NL
	 IF (SIOUT(M).EQ.K) THEN
	  WRITE(6,615)K,M
	  DO 361 I=1,NY
	   DO 362 J=1,NX
	    SP(I,J,K)=SP(I,J,K)*HNDM
 362       CONTINUE
 361      CONTINUE
	  DO 363 I=1,NY
           WRITE(14,991)(SP(I,J,K),J=1,NX)
 363      CONTINUE
	  DO 365 I=1,NY
	   DO 366 J=1,NX
	    SP(I,J,K)=SP(I,J,K)/HNDM
 366       CONTINUE
 365      CONTINUE
	  M=M+1
	 END IF
 360    CONTINUE
C********************************************************
 210    CONTINUE
 991	FORMAT(13F10.2)
  	RETURN
	END	




















