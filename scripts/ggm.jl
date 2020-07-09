using Plots
using Distributions
using LinearAlgebra
using LaTeXStrings
using Flux

## Generate data,
μ1 = [-3.];
σ1 = [2.];
μ2 = [3.];
σ2 = [1.];
w1 = [0.8];
μ3 = [-4.];
σ3 = [1.];
μ4 = [5.];
σ4 = [1.];
w2 = [0.4];
λ1 = 20.
λ2 = 10.

M1=MixtureModel([Normal(μ1[1],σ1[1]),Normal(μ2[1],σ2[1])],[w1[1], (1. -w1[1])])
x = rand(M1,300)
M2=MixtureModel([Normal(μ3[1],σ3[1]),Normal(μ4[1],σ4[1])],[w2[1], (1. -w2[1])])
y = rand(M2,300)
# reimplement for zygote
logpdfM1(x) = log( w1[1]/(sqrt(2*pi)*σ1[1])*exp(-0.5*((x-μ1[1])/σ1[1])^2) +
               (1. -w1[1])/(sqrt(2*pi)*σ2[1])*exp(-0.5*((x-μ2[1])/σ2[1])^2)
)
logpdfM2(y) = log( w2[1]/(sqrt(2*pi)*σ3[1])*exp(-0.5*((y-μ3[1])/σ3[1])^2) +
               (1. -w2[1])/(sqrt(2*pi)*σ4[1])*exp(-0.5*((y-μ4[1])/σ4[1])^2)
)

loss1(x) = -sum(logpdfM1.(x))
loss2(y) = -sum(logpdfM2.(y))
p_hx=Plots.histogram(x,normalize=:pdf, nbins = 100, xlabel="x", ylabel = "normalized counts", label="normalized histogram" );
p_hy=Plots.histogram(y,normalize=:pdf, nbins = 100,, xlabel="y", ylabel = "normalized counts",  label="normalized histogram" )

P1 = Poisson(λ1);
P2 = Poisson(λ2);
z = rand(P1, 300);
q = rand(P2, 300);
fit_mle(Poisson, z);
fit_mle(Poisson, q);


plot!(p_hx,-10:0.1:7,x->exp(logpdfM1(x)),label="true",linewidth = 4, color =:red);
plot!(p_hy,-8:0.1:7,y->exp(logpdfM2(y)),label="true",linewidth = 4);
niter = 1000;
ps1 = Flux.params([μ1,σ1,μ2,σ2,w1]);
ps2 = Flux.params([μ3,σ3,μ4,σ4,w2]);

opt = ADAM(0.01);
# for DEBUG
# Pa=zeros(5,niter);
# LL=zeros(niter);
# for i=1:niter
#     global Pa
#     gs = gradient(()->loss(x),ps);
#     Flux.update!(opt,ps,gs);
#     Pa[:,i]=reduce(vcat,ps)
#     LL[i]=loss(x)
# end

mydata=Iterators.repeated((x,),100);
Flux.train!(loss1,ps1,mydata,ADAM(0.1));
mydata=Iterators.repeated((y,),100);
Flux.train!(loss2,ps2,mydata,ADAM(0.1));

plot!(p_hy,-8:0.1:8,x->exp(logpdfM2(x)),label="estimated", linewidth = 4)
plot!(p_hx,-10:0.1:7,x->exp(logpdfM1(x)),label="estimated", linewidth = 4, color =:lightgreen)
[μ1 σ1 μ2 σ2 w1; μ3 σ3 μ4 σ4 w2]


