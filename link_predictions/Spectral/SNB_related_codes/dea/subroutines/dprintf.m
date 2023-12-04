function dprintf(mystr)
%DPRINTF debug print
    %% debug output

    %debugflag=false;
    debugflag=true;
    if(debugflag)
        fprintf(mystr);
    end

end

