a=E(idx,1);
b=E(idx,2);
qa=ceil(a/q);
qb=ceil(b/q);
begina=(qa-1)*q+1;
enda=qa*q;
beginb=(qb-1)*q+1;
endb=qb*q;
good=false;
while ~good
    good=true;
    bb=randi([beginb endb],1,1);
    alist=find(A(bb,:));
    if(length(alist)<K-0.1)
        good=false;
        continue;
    end
    idxaa=randi([1 length(alist)],1,1);
    aa=alist(idxaa);    %find <aa bb> that already existing.
    if(aa==bb)
        good=false;
        continue;
    end
    tmpa=find(A(a,:) == bb);
    if(~isempty(tmpa))
        good=false;
        continue;
    end
    tmpa=find(A(b,:) == aa);
    if(~isempty(tmpa))
        good=false;
        continue;
    end
end
%[aa bb]
idx2=find(logical( (E(:,1)==aa) .* (E(:,2)==bb) ));
if(isempty(idx2))
    idx2=find(logical( (E(:,1)==bb) .* (E(:,2)==aa) ));
end
assert(~isempty(idx2));
E(idx,2)=bb;
E(idx,1)=a;
E(idx2,1)=aa;
E(idx2,2)=b;
A(a,b)=0;
A(b,a)=0;
A(aa,bb)=0;
A(bb,aa)=0;
A(a,bb)=1;
A(bb,a)=1;
A(aa,b)=1;
A(b,aa)=1;
