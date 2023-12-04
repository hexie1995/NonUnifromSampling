function rm_files(basename)
    app={'.am','.aue','.cab','.conf','.na','.spm'};
    for i=1:6
        cmd=sprintf('/bin/rm %s%s',basename,app{i});
        system(cmd);
    end
end
