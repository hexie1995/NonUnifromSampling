% This demo generates a network by Block model, compute its spectrum and
% display the full spectrum.
    path(path,'../../');
    path(path,0'../../subroutines/');

    numvec=-2; 
    mode=5;
    ovl_norm=1;
    
    plot_circle_inverse=0;
    n=1000;
    c=3;
    epsilon=0.1;
    seed=2;
    %compute full spectrum of DEA matrix
    re=deaspec('gen_flag',1,'n',n,'c',c,'epsilon',epsilon,'seed',seed,'ovl_norm',ovl_norm,'mode',mode,'numvec',numvec,'do_clustering',0,'cbegin',1,'cend',2);
    D=re.D;
    V=re.V;
    a=real(D);
    b=imag(D);
    in=0.05;
    if(length(re.ovl))>1
        ovl=re.ovl(2);
    else
        ovl=re.ovl(1);
    end
    
    %% real values
    xr=D( logical((abs(imag(D))<0.01) .* ~((real(D)>=0.99) .* (real(D)<=1.01))   .* ~((real(D)<=-0.99) .* (real(D)>=-1.01)) .* ~((real(D)<=0.01) .* (real(D)>=-0.01)) ));
    %% plot distribution
    close all;
    hFig = figure(1);
    shiftaxis=1;
    cc=a(1);
    plot(a,b,'r.','markersize',8,'linewidth',2);
    hold on;
    plot(a(logical((abs(imag(D))<0.000001))),b(logical((abs(imag(D))<0.000001))),'r.','markersize',25,'linewidth',2);

    %draw cicle
    x=[-sqrt(cc+0.1):0.001:sqrt(cc+0.1)];
    y=sqrt(cc-x.^2);
    set(gca,'ytick',[-2 -1 1 2]);
    set(gca,'xtick',[-1 1 2 3]);
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
    txt1=sprintf('Overlap: %0.4f', ovl);
    annotation('textbox',[0.58 0.702 0.35 0.222],'String',{txt1},...
    'FitBoxToText','off',...
    'LineStyle','none','fontsize',30);
    set(gca,'fontsize',20);
    
    %%shift the axis
    if(shiftaxis)
        X=get(gca,'Xtick');
        Y=get(gca,'Ytick');

        % GET LABELS
        XL=get(gca,'XtickLabel');
        YL=get(gca,'YtickLabel');
        XL(XL=='0')=' ';
        YL(YL=='0')=' ';
        
        % GET OFFSETS
        Xoff=diff(get(gca,'XLim'))./60;
        Yoff=diff(get(gca,'YLim'))./60;

        % DRAW AXIS LINEs
        plot(get(gca,'XLim'),[0 0],'k','linewidth',1.);
        plot([0 0],get(gca,'YLim'),'k','linewidth',1.);

        % Plot new ticks  
        for i=1:length(X)
            plot([X(i) X(i)],[0 Yoff],'-k');
        end;
        for i=1:length(Y)
           plot([Xoff, 0],[Y(i) Y(i)],'-k');
        end;
        
        % ADD LABELS
        text(X,zeros(size(X))-3*Yoff,XL,'fontsize',20);
        text(zeros(size(Y))-3*Xoff,Y,YL,'fontsize',20);

        box off;
        % axis square;
        axis off;
        set(gcf,'color','w');
    end
    
    x=get(gca,'xlim');
    xlen=max(x)-min(x);
    y=get(gca,'ylim');
    ylen=max(y)-min(y);
    set(gcf,'PaperPositionMode','auto');
    set(hFig, 'Position', [1000 1000 700 700/xlen*ylen]);
