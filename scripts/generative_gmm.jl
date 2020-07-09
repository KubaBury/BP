using Plots
using Distributions
using LinearAlgebra
using LaTeXStrings


## Generate data,
μ1 = [-3.];
σ1 = [2.];
μ2 = [3.];
σ2 = [1.];
w1 = [0.7];

M=MixtureModel([Normal(μ1[1],σ1[1]),Normal(μ2[1],σ2[1])],[w1[1], (1. -w1[1])])
x = rand(M,300)

# reimplement for zygote
logpdfM(x) = log( w1[1]/(sqrt(2*pi)*σ1[1])*exp(-0.5*((x-μ1[1])/σ1[1])^2) +
               (1. -w1[1])/(sqrt(2*pi)*σ2[1])*exp(-0.5*((x-μ2[1])/σ2[1])^2)
)
loss(x) = -sum(logpdfM.(x))
p_hx=Plots.histogram(x,normalize=:pdf, nbins = 100,label="normalized histogram", xlabel="x", ylabel = "normalized counts" );
display(p_hx)
plot!(-7:0.1:7,x->exp(logpdfM(x)),label="true")

niter = 1000;
ps = params([μ1,σ1,μ2,σ2,w1]);
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

mydata=Iterators.repeated((x,),100)
Flux.train!(loss,ps,mydata,ADAM(0.1))

plot!(-7:0.1:7,x->exp(logpdfM(x)),label="estimated")
