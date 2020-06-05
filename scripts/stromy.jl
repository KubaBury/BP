using Distributions
using Statistics
using Plots
using LaTeXStrings


i = 200 #celkovy pocet y_i
λ = 10 #rate in Poisson distribution
P = Poisson(λ)
U = Uniform(0,10)
pocet_x = rand(P,i) #pocty x_k prirazene k jednomu y_i


#p(y,\bar{x})=p(\bar{x})*p(y|\bar{x})
xx = [rand(U, pocet_x[j]) for j =1:i]
mx = mean.(xx)
ms = std(hcat(mx...))
p_mx=Plots.histogram(mx, nbins = 30, xlabel=L"\bar{x}", ylabel = "counts" );
display(p_mx)
y = 1 .+ mx .+ mx.^2 .+ randn(length(mx))
cn = scatter(mx,y, markersize = 1.5, label = "data", legend =:topleft, xlabel=L"\bar{x}", ylabel = "y")
W = [(mx).^i for i = 0:2];
X = hcat(W...);
θ = inv(X'*X)*X'*y;
plt(k,l) = pdf.(Normal.([1 k k^2]*θ, 1. ), l) .* pdf.(Normal.(mean(mx), ms), k) 
contour!(cn, 2.5:0.1:7.5,10:0.1:60, (k,l)->plt(k,l)[1], xlabel=L"\bar{x}", ylabel=L"y", levels=6, )
p_y=Plots.histogram(y, nbins = 100, xlabel=L"\bar{x}", ylabel = "counts" );
display(p_y)
