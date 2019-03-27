function X = corrupt(X,rate)

[m,n]=size(X);
rate=floor(n*rate/100);


for i=1:m
     selected=randperm(n,rate);
     X(i,selected)=0;%rand(rate,1); 
end
end