%This demo displays the full spectrum of DEA matrix of various real
%networks.
path(path,'../../');
path(path,'../../subroutines/');


%gmlfname='karate.gml'; % file name of the network
%gmlfname='dolphins.gml'; % file name of the network
%gmlfname='polblogs.gml'; % file name of the network, note that only few
%eigenvalues are plotted, since computing full spectrum of the grapa is too
%slow.
gmlfname='polbooks.gml'; % file name of the network
%gmlfname='football.gml'; % file name of the network
%gmlfname='adjnoun.gml'; % file name of the network

[E,sigma]=read_gml(gmlfname);
m=length(E);% system size
if(m>1000)
    numvec=200; % if the networks is too large, compute first 200 eigenvectors.
else
    numvec=-2;  % if the networks is not too lage, compute full spectrum.
end
q=max(sigma); %number of groups in true configuration
mode=5; % DEA matrix
%mode=5; % matrix F, which is B' in paper.
if(q>2)
    do_clustering=3; % if q>2, use k-means clustering
else
    do_clustering=0; % if q==2, use sign of second largest real eigenvector
end
re=deaspec('ovl_norm',1,'q',q,'mode',mode,'numvec',numvec,'ftype','gml','basename',gmlfname,'do_clustering',do_clustering,'cbegin',1,'cend',q,'lcc',1);

D=re.D;
V=re.V;
if(length(re.ovl)>=2)
    ovl=re.ovl(2);
else
    ovl=re.ovl(1);
end

a=real(D);
b=imag(D);
in=0.05;

%% real values
x=D( logical((abs(imag(D))<0.01) .* (real(D)>0.5) .* ~((real(D)>=0.99) .* (real(D)<=1.01))));
x1=sort(x);
xr=D( logical((abs(imag(D))<0.01) .* ~((real(D)>=0.99) .* (real(D)<=1.01))   .* ~((real(D)<=-0.99) .* (real(D)>=-1.01)) .* ~((real(D)<=0.01) .* (real(D)>=-0.01)) ));
Dreal=D( logical((abs(imag(D))<0.000001)));
Dreal=unique(Dreal);
[~,idx]=sort(abs(Dreal),'descend');
Dreal=Dreal(idx);
%% plot distribution
close all;
hFig = figure(1);
shiftaxis=1; %shift axis to origin in plot
%draw eigenvalues in a complex plane
cc=a(1); %largest eigenvalue
plot(a,b,'r.','markersize',15,'linewidth',2);
hold on;
plot(a(logical((abs(imag(D))<0.000001))),b(logical((abs(imag(D))<0.000001))),'r.','markersize',25,'linewidth',2);
%draw a cycle
x=[-sqrt(cc+0.1):0.001:sqrt(cc+0.1)];
y=sqrt(cc-x.^2);
hold on;
plot(real(x),real(y),'b-','linewidth',1.);
plot(real(x),-real(y),'b-','linewidth',1.);

%eigenvalues
x=get(gca,'xlim');
y=[0,0];
plot(x,y,'black--','linewidth',0.5)
E=re.E;
A=A2E(E);
N=length(A);
di=sum(A);
c=full(mean(di));
xr=unique(xr);
xr = xr(xr>sqrt(cc));
fprintf('number of eigenvalues on real axes out of circle is %d\n',numel(xr));
txt1=sprintf('%s \nq=%d \nOverlap: %0.4f', gmlfname,q,ovl);
annotation('textbox' ,[0.58 0.702 0.35 0.222],'String',{txt1},...
'FitBoxToText','off',...
'LineStyle','none','fontsize',20);
set(gca,'fontsize',20);
%%shift the axis
% if(shiftaxis)
%     X=get(gca,'Xtick');
%     Y=get(gca,'Ytick');
% 
%     % GET LABELS
%     XL=get(gca,'XtickLabel');
%     YL=get(gca,'YtickLabel');
%     for i=1:length(XL)
%         if(XL(i)=='0')
%             XL(i)=' ';
%         end
%     end 
%     for i=1:length(YL)
%         if(YL(i)=='0')
%             YL(i)=' ';
%         end
%     end
% 
%     % GET OFFSETS
%     Xoff=diff(get(gca,'XLim'))./60;
%     Yoff=diff(get(gca,'YLim'))./60;
% 
%     % DRAW AXIS LINEs
%     plot(get(gca,'XLim'),[0 0],'k','linewidth',1.);
%     plot([0 0],get(gca,'YLim'),'k','linewidth',1.);
% 
%     % Plot new ticks  
%     for i=1:length(X)
%         plot([X(i) X(i)],[0 Yoff],'-k');
%     end;
%     for i=1:length(Y)
%        plot([Xoff, 0],[Y(i) Y(i)],'-k');
%     end;
% 
%     % ADD LABELS
%     text(X,zeros(size(X))-3*Yoff,XL,'fontsize',20);
%     text(zeros(size(Y))-3*Xoff,Y,YL,'fontsize',20);
% 
%     box off;
%     % axis square;
%     axis off;
%     set(gcf,'color','w');
%     
%     
% end    
x=get(gca,'xlim');
xlen=max(x)-min(x);
y=get(gca,'ylim');
ylen=max(y)-min(y);
set(gcf,'PaperPositionMode','auto');
set(hFig, 'Position', [1000 1000 700 700/xlen*ylen]);
