module fun_cch2
export cch2

function cch2(x, v1, v2, p)
    (w, v, dw) = chf1(x, v1, v2)
    rr = dw./w
    if p == 0
        y = real(rr)
    else
        y = (rr - real(rr))/sqrt(-1 + 0im)
    end
    return y
end

end