module fun_chf1
export chf1

function chf1(t2, v1, v2)
    n = size(v1)[1]
    k1 = size(v1)[2]
    k2 = size(v2)[2]
    s2 = size(t2)

    if s2[1] > s2[2]
        t2 = transpose(t2)
        d = s2[1]
    else 
        d = s2[2]
    end

    w = 0
    v = 0
    dw = 0
    g1 = ones(1, d)
    g2 = ones(n, 1)
    l1 = 1

    while l1 <= k1
        l2 = 1
        while l2 <= k2
            w0 = sum(exp(sqrt(-1 + 0im)*kron(g2,t2).*kron(g1,v2[:,l2])), dims=1)
            v0 = sum(exp(sqrt(-1 + 0im)*kron(g2,t2).*kron(g1,v1[:,l1])), dims=1)
            dw0 = sum(sqrt(-1 + 0im)*kron(g1,v1[:,l1]).*exp(sqrt(-1 + 0im)*kron(g2,t2).*kron(g1,v2[:,l2])), dims=1)
            dw = dw .+ dw0
            w = w .+ w0
            v = v .+ v0
            l2 = l2 + 1
        end
        l1 = l1 + 1
    end

    w = w/(n*k1*k2)
    v = v/(n*k1*k2)
    dw = dw/(n*k1*k2)

    if s2[1] > s2[2]
        w = transpose(w)
        v = transpose(v)
        dw = transpose(dw)
    end

    return [w, v, dw]
end

end