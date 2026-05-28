	PROGRAM pvpialln

c  ** modified to read unformatted output from nmc tape, (by newread1600wu)

c  ** THIS IS THE MODIFICATION OF CHRIS DAVIS'S PROGRAM PVPI.FOR
C  ** IT CALCULATES PV ON PI SURFACE AND INVERT THE STRAM FUNCTION FROM
C  **  RELATIVE VORTICITY.
c  **  THE NEW PART IS THAT IT TAKES SEVERAL HALFDAYS OF DATA AND
C  *8  CALCULATE THE MEAN TOO.
C  **  FOUR OUTPUT FILES, 1. THB THT, AND Q,  2. H, Psi, 
C  **      3. THBM, THTM AND QM,  4. HM. AND PSIM   WILL BE USED AS 
C  **  THE INPUT FOR  THE PROGRAM "QINVERTPNEW.FOR"       9/27/91  C.-C. WU

	PARAMETER (NW=10)
        PARAMETER (NY=21)
        PARAMETER (NX=45)
	REAL U(NY,NX,NW),V(NY,NX,NW),TH(NY,NX,NW),Q(NY,NX,NW),
     +	     FC(NY),AP(NY),APM(NY),APP(NY),MI,LL,PII,HDR(8),VB(NX,4),
     +	     VOR(NY,NX,NW),STB(NY,NX,NW),DUDY,DVDX,DTHX(NY,NX,NW),
     +	     DU(NY,NX,NW),DV(NY,NX,NW),PI(10),KAP,COEF,DEL,BETA,
     +	     DTHY(NY,NX,NW),VORA,DEF,PR(NW),PIB,PIT,
     +	     THB(NY,NX),THT(NY,NX),H(NY,NX,NW),ZHDR(8) 

C  **  DECLARE THE "A"                   9/27/91   WU
       
        REAL A(NY,5), PSI(NY,NX,NW),gg,sigm,Lapsi,Rs, Thrs,omegs

c  ******* Declare the mean variable,    10/22/91   wu

        real Thbm(NY,NX),thtm(NY,NX),Qm(NY,NX,NW)
     *       ,Hm(NY,NX,NW),Psim(NY,NX,NW)

c  ******************************

        Real Scale
        Integer imax, icount

c  ******* The information about the total numbers of halfdays and
c  ******  the output halfday number we want   10/22/91   wu

        Integer nhalfday,nhalfdayo,nhalfout(30),ic

c  **********************************
        LOgical IT
                           
C  *******************

	CHARACTER*80 F(50),INFILES(50)
	DATA PR/ 1.0, 0.85, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1/

C  **  add F(3) that writes the output for H and Psi   Wu 9/30/91

        scale=1.
   
c	TYPE*,'FILES(1)? IN=(H,TH,U,V)'
c	READ (5,'(a)')F(1)

	TYPE*,'FILES(2)? OUT=(Thbm,thtm and Qm)'
	TYPE*,'FILES(3)? OUT=(Hm and Psim)'

	READ (5,'(a)')(F(I),I=2,3)
       

c  ** read the total number of input half days  **

        write (6,*) 'total numbers of input halfdays?'
        read (5,*) nhalfday
        write (6,*) 'names of input files?'
	READ (5,'(a)')(INFILES(I),I=1,nhalfday)
	OPEN(7,FILE=INFILES(1),STATUS='OLD',form='formatted')
	READ(7,*)(HDR(J),J=1,8)

	OPEN(11,FILE=F(2),STATUS='new')
	WRITE(11,*)(HDR(J),J=1,8)
	OPEN(12,FILE=F(3),STATUS='new')
	WRITE(12,*)(HDR(J),J=1,8)



c  ** read the total numbers the output halfdays

        type *, 'total number of output halfdays :(nhalfdayo)'
         read (5,*) nhalfdayo

         do 888 nn=1,nhalfdayo

c  ** read the output halfday

        type *, 'the output halfday'
         read (5,*) nhalfout(nn)

	TYPE*,'FILES(4+2*(nn-1))? OUT=(Thb,tht,Q )'
	TYPE*,'FILES(5+2*(nn-1))? OUT=(H and psi)'

	READ (5,'(a)')(F(I+2*(nn-1)),I=4,5)

c        TYPE*,'SCALE FOR INPUT?'
c        READ*,SCALE	

	OPEN(13+2*(nn-1),FILE=F(4+2*(nn-1)),STATUS='new')
	WRITE(13+2*(nn-1),*)(HDR(J),J=1,8)
	OPEN(14+2*(nn-1),FILE=F(5+2*(nn-1)),STATUS='new')
	WRITE(14+2*(nn-1),*)(HDR(J),J=1,8)
       
c  *******************************************

888      continue

c  ** read the information for the realtive vorticity inversion
   
c        read (5,*)imax
c        read (5,*)omegs
c        read (5,*)thrs
       imax=300
       omegs=1.75
       thrs=5.e4

c*********************

	PII=4.*ATAN(1.)
	CP=1004.5
	P0=1.E5
	KAP=2./7.
	MI=9999.90
	AA=2.E7/PII
   
c  **  put gravity       WU 9/30/91

        gg=9.8066
        SIGM=Hdr(5)/Hdr(6)

c  ***********************************

	DL=AA*HDR(5)*PII/180.
	DP=AA*HDR(6)*PII/180.
  	DO K=1,NW
	 PI(K)=CP*( PR(K)**KAP )
	END DO
	PIB=0.5*(PI(2)+PI(1))
	PIT=0.5*(PI(NW)+PI(NW-1))

 	DO I=1,HDR(8)

	 AP(I)=COS( PII*( HDR(3) - (I-1)*HDR(6) )/180. )
	 APM(I)=COS( PII*( HDR(3) - (I-1.5)*HDR(6) )/180. )
	 APP(I)=COS( PII*( HDR(3) - (I-0.5)*HDR(6) )/180. )

C ** ADD THE LAPLACIAN COEFFICIENT HERE          9/28/91  WU

         A(I,1)=SIGM*SIGM*APM(I)/AP(I)
         A(I,2)=1./( AP(I)*AP(I) )
         A(I,3)=-( 2. + SIGM*SIGM*AP(I)*(APM(I)+APP(I)) )/
     +      ( AP(I)*AP(I) )
         A(I,4)=1./( AP(I)*AP(I) )
         A(I,5)=SIGM*SIGM*APP(I)/AP(I)

c         print *, 'a(i,4) = ', a(i,4)
c         print *, 'a(i,5) = ', a(i,5)
 
C *****************************************

	 FC(I)=(1.458E-4)*SIN( PII*( HDR(3) - (I-1)*HDR(6) )/180. )

         write(6,*) i, HDR(3) - (I-1)*HDR(6), fc(i)

	END DO
                                               
c  **  MAKE THE MEANBE ZERO INITIALLY  
                                
          DO J=1, HDR(7)
             DO I=1, HDR(8)
                THBM(I,J)=0.
                THTM(I,J)=0.
                DO K=1,NW
                   QM(I,J,K)=0.
                   HM(I,J,K)=0.
                   PSIM(I,J,K)=0.
                END DO
             END DO
          END DO
                

C       BEGIN DO LOOP FOR HALFDAYS


	DO K=2,NW-1
	 DO I=1,HDR(8)
	  Q(I,1,K)=MI
	  Q(I,HDR(7),K)=MI
	 END DO
	 DO J=1,HDR(7)
	  Q(1,J,K)=MI
	  Q(HDR(8),J,K)=MI
	 END DO
 	END DO

         write(6,*)'hdr',hdr

        do 1000 ihalfday=1, nhalfday

           print *, ' nhalfday= ', ihalfday
           if (ihalfday.ne.1) then
	     OPEN(7,FILE=INFILES(ihalfday),STATUS='OLD',form='formatted')
	     READ(7,*)(ZHDR(J),J=1,8)
             do ii=1,8
               if (HDR(ii).ne.ZHDR(ii)) then
                 write(6,*) 'Input data file headers are inconsistent'
                 stop
               endif
             enddo
           endif
              
  	DO K=1,NW
c	 WRITE(6,*)K
	 DO I=1,HDR(8)
  	  READ(7,91)(H(I,J,K),J=1,HDR(7))
c  	  READ(7,111,END=2)(H(I,J,K),J=1,HDR(7))
c111        format (17f7.1)
c  2	  DO J=1,HDR(7)
c	   H(I,J,K)=SCALE*H(I,J,K)
   91 FORMAT(10F8.1) 

c **  let initial psi equal H,    Wu 9/30/91

c           psi(i,j,k)=H(i,j,k)*gg/fc(i)

C ********************************

c	  END DO
	 END DO
	END DO
        write(6,*) 'read heights'
  	DO K=1,NW
	 DO I=1,HDR(8)
  	  DO J=1,HDR(7)

c **  let initial psi equal H,    Wu 9/30/91

           psi(i,j,k)=H(i,j,k)*gg/fc(i)

C ********************************

	  END DO
	 END DO
	END DO

c        print *, 'psi(5,5,2) = ' , psi(5,5,2)
c        print *,'psi(1,1,1) = ', psi(1,1,1), psi(1,2,1)       

  	DO K=1,NW
  	 DO I=1,HDR(8)
    	  READ(7,91)(TH(I,J,K),J=1,HDR(7))
c  	  READ(7,111,END=3)(TH(I,J,K),J=1,HDR(7))
c  3	  DO J=1,HDR(7)

C  **  CALCULATE THE POTENTIAL TEMPERATURE

C	   TH(I,J,K)=TH(I,J,K) * pr(k) **(- .287)
c	  END DO
	 END DO
	END DO

        write(6,*) 'read temps, hdr7, hdr8 ', hdr(7), hdr(8)

  	DO K=1,NW
  	 DO I=1,HDR(8)
       	   DO J=1,HDR(7)

C  **  CALCULATE THE POTENTIAL TEMPERATURE

c   **  the data is already in K ,  3/3/92  wu

 	    TH(I,J,K)=TH(I,J,K)*CP/PI(K)
	   END DO
	 END DO
	END DO

  	DO K=1,NW
	 DO I=1,HDR(8)
  	  READ(7,91)(U(I,J,K),J=1,HDR(7))
c  	  READ(7,111,END=4)(U(I,J,K),J=1,HDR(7))
c  4	  DO J=1,HDR(7)
c	   U(I,J,K)=SCALE*U(I,J,K)
c	  END DO
	 END DO
	END DO
        write(6,*) 'read u'

    	DO K=1,NW
  	 DO I=1,HDR(8)
  	  READ(7,91)(V(I,J,K),J=1,HDR(7))
c  	  READ(7,111,END=5)(V(I,J,K),J=1,HDR(7))
c  5	  DO J=1,HDR(7)
c	   V(I,J,K)=SCALE*V(I,J,K)
c	  END DO
	 END DO
	END DO
        close(7)
        write(6,*) 'read v'

C***************************************************************
 	DO J=1,HDR(7)
  	 DO I=1,HDR(8)
	  THB(I,J)=0.5*(TH(I,J,1) + TH(I,J,2)) 
	  THT(I,J)=0.5*(TH(I,J,NW-1) + TH(I,J,NW))
	 END DO
	END DO
C*******************************************************************
        DO K=2,NW-1
         DO J=2,HDR(7)-1
          DO I=2,HDR(8)-1        
	   DTHY(I,J,K)=(TH(I-1,J,K)-TH(I+1,J,K))/(2.*DP)
	   DTHX(I,J,K)=(TH(I,J+1,K)-TH(I,J-1,K))/(2.*AP(I)*DL)
	   STB(I,J,K)=(TH(I,J,K+1)-TH(I,J,K-1))/
     +	     (PI(K+1)-PI(K-1))
	  END DO
	 END DO
	END DO
C************** Vorticity *******************
	DO K=1,NW
	 DO I=2,HDR(8)-1
	 DO J=2,HDR(7)-1
	  VL=(V(I,J+1,K)-V(I,J-1,K))/(2.*DL*AP(I))
	  UPV=(AP(I-1)*U(I-1,J,K)-AP(I+1)*U(I+1,J,K))/
     +	       (2.*DP*AP(I))
	  VOR(I,J,K)=VL - UPV
	 END DO
	 END DO
	END DO

C  ** TO INVERT THE STREAM FUNCTION (PSI) FROM RELATIVE VORTICITY(VOR)
C  **                                           9/27/91  WU

c  **  solve the lateral psi
c  **  calculate the boundary divergence term

       Do k=1,NW
          dsum=0.

       do i=1,Hdr(8)-1
          dsum=dsum -((u(i,1,k)+u(i+1,1,k))/2.)* DP
          dsum=dsum +((u(i,Hdr(7),k)+u(i+1,Hdr(7),k))/2.)* DP
       End do

       do j=1,Hdr(7)-1
          dsum=dsum+((v(1,j,k)+v(1,j+1,k))/2.)* DL * AP(1)
          dsum=dsum-((v(Hdr(8),j,k)+v(Hdr(8),j+1,k))/2.)
     *                 *DL*AP(Hdr(8))
       End do

       dsum = dsum /( 2.*DP*(Hdr(8)-1)+ DL*(HDr(7)-1)
     *               *(Ap(1)+Ap(Hdr(8)))  )
       
       print *, 'dsum=', dsum

c  ** choose starting point psi(1,1,k)=H(1,1,k)*g/f             
       
c  **  then integrate by Davis (2.40) to get the whole psi      

       do i=1,Hdr(8)-1
          psi(i+1,1,k)=psi(i,1,k)+(dsum+((u(i  ,1,k)
     *                           +        u(i+1,1,k) )/2.))* DP
       End do

       do j=1,Hdr(7)-1
          psi(HDR(8),j+1,k)=psi(hdr(8),j,k)+
     *    ( dsum+((v(Hdr(8),j,k)+v(Hdr(8),j+1,k))/2.) )
     *                 *DL*AP(Hdr(8))
       End do

       do i=Hdr(8),2,-1
          psi(i-1,Hdr(7),k)=psi(i,Hdr(7),k)
     *     +(dsum-((u(i,Hdr(7),k)+u(i-1,Hdr(7),k) )/2.))* DP
       End do

       do j=Hdr(7),3,-1
          psi(1,j-1,k)=psi(1,j,k)+
     *    ( dsum-((v(1,j,k)+v(1,j-1,k))/2.) )
     *                 *DL*AP(1)
       End do

       End do

c        print *,'psi = ', psi(1,1,1), psi(1,2,1)       

c  **  invert to get the interior psi by overrelaxation

       icount=0
       
c       imax=200
c       omegs=1.75
c       thrs=1.e3

155    it=.true.    

       do k=1,NW
       do i=2,HDR(8)-1
       do j=2,HDR(7)-1

       Lapsi=1./(DL*DL)*( A(I,1)*psi(I-1,J,K) + A(I,2)*psi(I,J-1,K)+       
     +      A(I,3)*psi(I,J,K) + A(I,4)*psi(I,J+1,K) +
     +      A(I,5)*psi(I+1,J,K) )
       RS = LApSi - vor(i,j,k)   
       
       If ((i.eq.5).and.(j.eq.10).and.(k.eq.3)) then
c          print *,'lapsi = ',lapsi,' vor = ', vor(i,j,k)
       endif

       psin=psi(i,j,k)
       psi(i,j,k)= psin-omegs*RS/A(i,3)*(DL*DL)

       Dpsi= Psi(i,j,k)-psin

       If (abs(dpsi).GT.thrs) then
          IT= .false.
          
       end if

       end do
       end do
       end do
       
       icount= icount + 1

c       if ((icount.eq.1).or.(icount.eq.10)) then 
c         print *, psi(5,10,3), H(5,10,3),u(5,10,3)
c         print *,dpsi, vor(5,10,3)
c       end if

c  **  check if too many iterations

       if (icount.gt. IMAX) THEN
          PRINT *, 'TOO MANY ITERATION FOR PSI'
          GO TO 123
       ENDIF

c  **  check if need to do more realxation

       If (IT) then
          print *, 'icount=', icount
          print *, 'psi converged'
       else
          go to  155       
       ENDIF   

123    continue
c       print *, 'i = ',i, ' j = ',j,' k = ',k

c       print *, 'psi(3,10,5) = ', psi(3,10,5)

C *******************************************************************
 
C******* Vertical wind shear *************************************
	DO K=2,NW-1
	 DO I=2,HDR(8)-1
	 DO J=2,HDR(7)-1
	  DU(I,J,K)=(U(I,J,K+1)-U(I,J,K-1))/(PI(K+1)-PI(K-1))
	  DV(I,J,K)=(V(I,J,K+1)-V(I,J,K-1))/(PI(K+1)-PI(K-1))
	 END DO
	 END DO
	END DO
C************ Calculate PV **************************************
	COEF=1.E2*1.E6*9.81*KAP*(CP**3.5)/P0
	WRITE(6,*)COEF
	DO L=2,NW-1
	 DO J=2,HDR(7)-1
	 DO I=2,HDR(8)-1
	  ZSHR=COEF*(PI(L)**-2.5)*( DU(I,J,L)*DTHY(I,J,L)
     +       - DV(I,J,L)*DTHX(I,J,L) )
	  Q(I,J,L)=-COEF*(PI(L)**-2.5)*( (FC(I)+VOR(i,j,L))*
     +		STB(I,J,L) ) - ZSHR
C************ Check for negative values ******
	  IF (Q(I,J,L).LE.0.) THEN
	      print *, 'Q is less than zero ', i,' * ', j,' * ', L
c              WRITE(6,102)Q(I,J,L),STB(I,J,L),V(I,J,L),ZSHR,I,J
 102	   FORMAT(4F10.3,2I3)
	  END IF
	 END DO
	 END DO
	END DO

C****** CHECK TO SEE IF WE ARE AT THE RIGHT DAY AND WRITE TO FILE
C****** IN ANY CASE AVERAGE THE FIELDS

         do 777 nn=1,nhalfdayo

        if( ihalfday.eq.nhalfout(nn)) then
           ic=nn
           go to 778
        end if

777      continue

         go to 779

C****** WRITE OUT "BOUNDARY" THETA AND PV ****************

778    	DO I=1,HDR(8)
	 WRITE(13+2*(ic-1),97)(THB(I,J),J=1,HDR(7))
	END DO
	DO I=1,HDR(8)
	 WRITE(13+2*(ic-1),97)(THT(I,J),J=1,HDR(7))
	END DO
	DO K=2,NW-1
	 DO I=1,HDR(8)
	  WRITE(13+2*(ic-1),97)(Q(I,J,K),J=1,HDR(7)) 
	 END DO
	END DO
  97	FORMAT(13F10.2)

c  **  write out H and psi  , WU 9/30/91  **

	DO K=1,NW
	 DO I=1,HDR(8)
	  WRITE(14+2*(ic-1),97)(H(I,J,K),J=1,HDR(7)) 
	 END DO
	END DO

	DO K=1,NW
	 DO I=1,HDR(8)

c  ** to scale psi to fit H

	  WRITE(14+2*(ic-1),97)(psi(I,J,K)/1.e5,J=1,HDR(7)) 
	 END DO                          
	END DO

c  ** write out u and v and theta
 	DO K=1,NW
	 DO I=1,HDR(8)
	  WRITE(14+2*(ic-1),97)(U(I,J,K),J=1,HDR(7)) 
	 END DO                          
	END DO
 	DO K=1,NW
	 DO I=1,HDR(8)
	  WRITE(14+2*(ic-1),97)(V(I,J,K),J=1,HDR(7)) 
	 END DO                          
	END DO
 	DO K=1,NW
	 DO I=1,HDR(8)
	  WRITE(14+2*(ic-1),97)(TH(I,J,K),J=1,HDR(7)) 
	 END DO                          
	END DO

         close (13+2*(ic-1))
         close (14+2*(ic-1))
        
c******* calculate the mean part

779	 DO I=1,HDR(8)
	 DO J=1,HDR(7)
	  THBm(I,J)=THBm(i,j)+THB(i,j)
	  THTm(I,J)=THTm(i,j)+THT(i,j)
	 END DO    
	 END DO

	DO K=2,NW-1
	 DO I=1,HDR(8)
	 DO J=1,HDR(7)
	  Qm(I,J,K)=Qm(i,j,k)+Q(i,j,k)
	 END DO
	 END DO
	END DO

	DO K=1,NW
	 DO I=1,HDR(8)
	 DO J=1,HDR(7)
	  Hm(I,J,K)=Hm(i,j,k)+H(i,j,k)
	  psim(I,J,K)=psim(i,j,k)+psi(i,j,k)
	 END DO
	 END DO
	END DO

1000    continue

c********  write out the mean output

C****** WRITE OUT "BOUNDARY" THETA AND PV ****************
	DO I=1,HDR(8)
	 WRITE(11,97)(THBm(I,J)/nhalfday,J=1,HDR(7))
	END DO
	DO I=1,HDR(8)
	 WRITE(11,97)(THTm(I,J)/nhalfday,J=1,HDR(7))
	END DO
	DO K=2,NW-1
	 DO I=1,HDR(8)
	  WRITE(11,97)(Qm(I,J,K)/nhalfday,J=1,HDR(7)) 
	 END DO
	END DO
  
c  **  write out H and psi  , WU 9/30/91  **

	DO K=1,NW
	 DO I=1,HDR(8)
	  WRITE(12,97)(Hm(I,J,K)/nhalfday,J=1,HDR(7)) 
	 END DO
	END DO

	DO K=1,NW
	 DO I=1,HDR(8)

c  ** to scale psi to fit H

	  WRITE(12,97)(psim(I,J,K)/(1.e5*nhalfday),J=1,HDR(7)) 
	 END DO
	END DO
        

c ****************************************

         close (7)                  
         close (11)
         close (12)

 	STOP
	END


