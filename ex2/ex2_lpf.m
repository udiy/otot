%File Name: ex2_lpf.m
%ID: 200434470
%Exercise 2
%DESCRIPTION: Low Pass Filter

function[ H, X, Y] = lpf_ex2(step, wc, M)

j=1;
%the etration between -pi & pi
for k=(-1*pi):step:pi
    %the x axis
    X(j)=k;
    %fourier
    Y(j)=wc/pi;
    for l=1:M
        Y(j)=Y(j)+(2*sin(wc*l)*exp(-1*1i*k*l))/(pi*l);
    end
    %building 'filter' shape
    if abs(k)<wc
        H(j)=1;
    else
        H(j)=0;
    end
    %advancing the point
    j=j+1;
end

end
