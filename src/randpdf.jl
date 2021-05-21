module fun_randpdf
export randpdf

function randpdf(p,px,dim)
    # Vectorization and normalization of the input pdf
    px=px[:]
    p=p[:]./trapz(px,p[:])

    # Interpolation of the input pdf for better integration with 10000 points
    pxi=[linspace(min(px),max(px),10000)]'
    pi=interp1(px,p,pxi,'linear')

    # computing the cumulative distribution function for input pdf
    cdfp = cumtrapz(pxi,pi)

    # finding the parts of cdf parallel to the X axis 
    ind=[true; not(diff(cdfp)==0)]

    # and cut out the parts
    cdfp=cdfp(ind)
    pi=pi(ind)
    pxi=pxi(ind)

    # generating the uniform distributed random numbers
    uniformDistNum=rand(dim)

    # and distributing the numbers using cdf from input pdf
    userDistNum=interp1(cdfp,pxi,uniformDistNum[:]','linear')

    # making graphs if no output exists
    if nargout==0
        subplot(3,4,[1 2 5 6])
        [n,xout]=hist(userDistNum,50)
        n=n./sum(n)./(xout[2]-xout[1])
        bar(xout,n)
        hold on
        plot(pxi, pi./trapz(pxi,pi),'r')
        hold off
        legend('pdf from generated numbers','input pdf')

        subplot(3,4,[3 4 7 8])
        plot(pxi, cdfp,'g')
        ylim([0 1])
        legend('cdf from input pdf')

        subplot(3,4,[9:12])
        plot(userDistNum)
        legend('generated numbers')
    else
        x = reshape(userDistNum,dim)
    end

    return x
end

end