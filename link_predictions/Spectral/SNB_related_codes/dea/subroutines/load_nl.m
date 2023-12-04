function [ E, ADJ, conf_true ] = load_nl(filename)
%LOAD_NL load neighbor list into a matrix
    fp = fopen(filename);
    tline = fgets(fp);
    [A count] = sscanf(tline,['%d' '%d']);
    numV=A(1,1);numE=A(2,1);
   
    
    ADJ=sparse(numV,numV);
    checkcount=0;
    
    for i=1:numV
         tline = fgets(fp);
         [A count] = sscanf(tline,['%d' '%d']);
         checkcount=checkcount+count;
         ADJ(i,A)=1;
    end
   
    if (numE~=checkcount/2)
        disp('Problem in reading the file');
    end
    fclose(fp);
    E=A2E(ADJ);
    conf_true=[];
end

