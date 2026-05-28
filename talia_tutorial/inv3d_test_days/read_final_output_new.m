% read output text files

clear all
close all

cd ~
cd /home/talia/inv3d_talia_test_days2


% fileID = fopen('03021200.grid');
% hh2= fscanf(fileID,'%f %f %f %f %f %f %f %f\n');
% hh3= fscanf(fileID,'%f');
% 
% fileID2 = fopen('03021200.grid');
% Aqq2= fscanf(fileID2,'%f',[4201 10]);

fileID = fopen('03021200.grid');
A=fscanf(fileID,'%f');


lat1=A(1);
lat2=A(3);
lon1=A(2);
lon2=A(4);
dx=A(5);
dy=A(6);
Nx=A(7);
Ny=A(8);

Nw=10;

AA=squeeze(A(9:end));
AAA=reshape(AA,Nx,Nw*Ny*4);

data=AAA';

h=zeros(Ny,Nx,Nw);
temp=zeros(Ny,Nx,Nw);
u=zeros(Ny,Nx,Nw);
v=zeros(Ny,Nx,Nw);

for j=1:Nw;
    h(:,:,j)=squeeze(data(Ny*(j-1)+1:Ny*(j-1)+Ny,:));
    temp(:,:,j)=squeeze(data(Ny*(j-1+Nw)+1:Ny*(j-1+Nw)+Ny,:));
    u(:,:,j)=squeeze(data(Ny*(j-1+2*Nw)+1:Ny*(j-1+2*Nw)+Ny,:));
    v(:,:,j)=squeeze(data(Ny*(j-1+3*Nw)+1:Ny*(j-1+3*Nw)+Ny,:));
end

z_level=5;

figure;
contourf(squeeze(h(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('Height ')
hold on; quiver(squeeze(u(2:end-1,2:end-1,z_level)),squeeze(-v(2:end-1,2:end-1,z_level)),'k')
axis ij

figure;
contourf(squeeze(temp(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('temp')
hold on; quiver(squeeze(u(2:end-1,2:end-1,z_level)),squeeze(-v(2:end-1,2:end-1,z_level)),'k')
axis ij


figure;
contourf(squeeze(u(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('u ')
hold on; quiver(squeeze(u(2:end-1,2:end-1,z_level)),squeeze(-v(2:end-1,2:end-1,z_level)),'k')
axis ij
%caxis([-20 70])


figure;
contourf(squeeze(v(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('v ')
hold on; quiver(squeeze(u(2:end-1,2:end-1,z_level)),squeeze(-v(2:end-1,2:end-1,z_level)),'k')
axis ij
%%%caxis([-50 50])

% delimiterIn = ' ';
% headerlinesIn = 2;
% A = importdata('03021200bal.out',delimiterIn,headerlinesIn);
% header1=A.textdata{1,1};
% header2=A.textdata{2,1};
% header_line1=str2num(header1);
% header_line2=str2num(header2);

%%

fileID2 = fopen('03021200pert.out');

A=fscanf(fileID2,'%f');

lat1=A(1);
lat2=A(3);
lon1=A(2);
lon2=A(4);
dx=A(5);
dy=A(6);
Nx=A(7);
Ny=A(8);

Nw=10;

n_inv=3;

AA=squeeze(A(9:end));
AAA=reshape(AA,Nx,Nw*Ny*2*3);


AAA_full=reshape(AA,Nx,Ny,Nw,2,3);   
AAA1=AAA_full(:,:,:,:,1);
AAA2=AAA_full(:,:,:,:,2);
AAA3=AAA_full(:,:,:,:,3);

data1=AAA1;
data2=AAA2;
data3=AAA3;

data=AAA';


% lat1=header_line1(1);
% lat2=header_line1(3);
% lon1=header_line1(2);
% lon2=header_line1(4);
% dx=header_line1(5);
% dy=header_line2(1);
% Nx=header_line2(2);
% Ny=header_line2(3);

% 
% data=A.data;


H=zeros(Ny,Nx,Nw);
psi=zeros(Ny,Nx,Nw);

for j=1:Nw;
    H(:,:,j)=squeeze(data(Ny*(j-1)+1:Ny*(j-1)+Ny,:));
    psi(:,:,j)=squeeze(data(Ny*(j-1+Nw)+1:Ny*(j-1+Nw)+Ny,:));
end

%pause 
%end

H1=zeros(Ny,Nx,Nw);
psi1=zeros(Ny,Nx,Nw);
H2=zeros(Ny,Nx,Nw);
psi2=zeros(Ny,Nx,Nw);
H3=zeros(Ny,Nx,Nw);
psi3=zeros(Ny,Nx,Nw);

H1=permute(data1(:,:,:,1),[2 1 3]);
psi1=permute(data1(:,:,:,2),[2 1 3]);
H2=permute(data2(:,:,:,1),[2 1 3]);
psi2=permute(data2(:,:,:,2),[2 1 3]);
H3=permute(data3(:,:,:,1),[2 1 3]);
psi3=permute(data3(:,:,:,2),[2 1 3]);



psi=psi2;
H=H2;

R_gas=287;
kappa=2/7;
Cp=R_gas./kappa;

PR= [1.0, 0.85, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1];

AA=2*10^7/pi;
DL=AA*dx*pi/180;
DP=AA*dy*pi/180;
for K=1:Nw
	 PI(K)=Cp*(PR(K).^kappa);
end

PIB=0.5*(PI(2)+PI(1));
PIT=0.5*(PI(Nw)+PI(Nw-1));

for I=1:Ny

AP(I)=cos(pi*(A(3) - (I-1)*A(6) )/180);
APM(I)=cos(pi*(A(3) - (I-1.5)*A(6) )/180);
APP(I)=cos(pi*( A(3) - (I-0.5)*A(6) )/180);

end

U=zeros(Ny,Nx,Nw);
V=zeros(Ny,Nx,Nw);

for K=1:Nw
 for I=2:Ny-1
     for J=2:Nx-1
	  psi_x3=(psi3(I,J+1,K)-psi3(I,J-1,K))/(2.*DL*AP(I));
      psi_y3=(psi3(I+1,J,K)-psi3(I-1,J,K))/(2.*DP);
	  U3(I,J,K)=(10^5)*psi_y3;
      V3(I,J,K)=(10^5)*psi_x3;
      
      psi_x1=(psi1(I,J+1,K)-psi1(I,J-1,K))/(2.*DL*AP(I));
      psi_y1=(psi1(I+1,J,K)-psi1(I-1,J,K))/(2.*DP);
	  U1(I,J,K)=(10^5)*psi_y1;
      V1(I,J,K)=(10^5)*psi_x1;
      
      psi_x2=(psi2(I,J+1,K)-psi2(I,J-1,K))/(2.*DL*AP(I));
      psi_y2=(psi2(I+1,J,K)-psi2(I-1,J,K))/(2.*DP);
	  U2(I,J,K)=(10^5)*psi_y2;
      V2(I,J,K)=(10^5)*psi_x2;
      
      
     end
 end
end


%%

U=U1;
V=V1;
psi=psi1;
H=H1;

figure;
contourf(squeeze(U(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('U induced')
hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
axis ij

figure;
contourf(squeeze(V(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('V induced')
hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
axis ij


% figure;
% contourf(squeeze(psi(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
% title('psi induced')
% hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
% axis ij
% 
% figure;
% contourf(squeeze(H(3:end-3,3:end-3,z_level)),60); shading flat; colorbar
% hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
% axis ij
% title('Height induced')

%%

U=U2;
V=V2;
psi=psi2;
H=H2;


figure;
contourf(squeeze(U(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('U induced')
hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
axis ij

figure;
contourf(squeeze(V(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('V induced')
hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
axis ij


% figure;
% contourf(squeeze(psi(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
% title('psi induced')
% hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
% axis ij
% 
% figure;
% contourf(squeeze(H(3:end-3,3:end-3,z_level)),60); shading flat; colorbar
% hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
% axis ij
% title('Height induced')

%%

U=U3;
V=V3;
psi=psi3;
H=H3;


figure;
contourf(squeeze(U(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('U induced')
hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
axis ij

figure;
contourf(squeeze(V(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
title('V induced')
hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
axis ij


% figure;
% contourf(squeeze(psi(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
% title('psi induced')
% hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
% axis ij
% 
% figure;
% contourf(squeeze(H(3:end-3,3:end-3,z_level)),60); shading flat; colorbar
% hold on; quiver(squeeze(U(2:end-1,2:end-1,z_level)),squeeze(-V(2:end-1,2:end-1,z_level)),'k')
% axis ij
% title('Height induced')


%% 

% figure;
% contourf(squeeze(UDIFF(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
% title('U DIFF at zlevel')
% axis ij
% hold on; quiver(squeeze(UDIFF(2:end-1,2:end-1,z_level)),squeeze(-VDIFF(2:end-1,2:end-1,z_level)),'k')
% %caxis([-10 10])
% 
% figure;
% contourf(squeeze(VDIFF(2:end-1,2:end-1,z_level)),60); shading flat; colorbar
% title('V DIFF at zlevel')
% axis ij
% hold on; quiver(squeeze(UDIFF(2:end-1,2:end-1,z_level)),squeeze(-VDIFF(2:end-1,2:end-1,z_level)),'k')
% %caxis([-10 10])

usum=squeeze(U1(2:end-1,2:end-1,z_level))+squeeze(U2(2:end-1,2:end-1,z_level))+squeeze(U3(2:end-1,2:end-1,z_level));
vsum=squeeze(V1(2:end-1,2:end-1,z_level))+squeeze(V2(2:end-1,2:end-1,z_level))+squeeze(V3(2:end-1,2:end-1,z_level));



figure;
contourf(usum,60); shading flat; colorbar
title('U sum')
hold on; quiver(usum,-vsum,'k')
axis ij
%caxis([-20 70])

figure;
contourf(vsum,60); shading flat; colorbar
title('V sum')
hold on; quiver(usum,-vsum,'k')
axis ij
%%%caxis([-50 50])









