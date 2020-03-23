using Flux
using Plots
using LaTeXStrings
using SpecialFunctions

function pd_ms(d,m,s)
    1/sqrt(s)*exp(-0.5*(d-m)^2/s)
end
function pm(m,τ,s)
    1/sqrt(τ*s)*exp(-0.5*m^2/(τ*s))
end
function ps(s)
    s^(-1)
end
joint(d1,d2,m,s,τ) = pd_ms(d1,m,s)*pd_ms(d2,m,s)*pm(m,τ,s)*ps(s);

plc=contour(1:0.01:4,0:0.01:0.5,(m,s)->joint(3.,2.,m,s,1000),
    xlabel=L"m", ylabel=L"\sigma", levels=5
)


# elbo
# qm = N(mm,sm) => 
# qs = iG(γ,δ) =>

#l_pd_ms(d,m,s) = -0.5log(s)-0.5*(d^2-2*d*m+m^2)/s;
El_pd_ms(d,mm,lsm,γ,δ) = -0.5*(log.(δ)-digamma.(γ)) .- 0.5*(d.^2 .- 2*d.*mm .+ mm.^2 .+ exp.(lsm)) .* (γ./δ);

#l_pm(m,τ,s) = -0.5(log(τ)+log(s))-0.5*m^2/(τ*s);
El_pm(mm,lsm,τ,γ,δ) = -0.5*(log.(τ) .+ (log.(δ)-digamma.(γ))) .- 0.5*(mm.^2 .+ exp.(lsm))./τ.*(γ./δ);

#l_ps(s) = -log(s)
El_ps(γ,δ) = -(log.(δ)-digamma.(γ));

#l_qm(m,mm,sm) = -log(sm)-0.5*(m-mm)^2/sm
El_qm(mm,lsm) = -lsm;

#l_gs = -
El_qs(γ,δ) = -(γ.+log.(δ.*gamma.(γ)).-(1 .+ γ).*digamma.(γ));

function elbo2(d,mm,lsm,γ,δ)
    sum(El_pd_ms(d[1],mm,lsm,γ,δ)+El_pd_ms(d[2],mm,lsm,γ,δ)+
    El_pm(mm,lsm,1000,γ,δ) + El_ps(γ,δ) - El_qm(mm,lsm)- El_qs(γ,δ))
end
mm = [1.0];
logsm = [1.0];
γ = [1.0];
δ = [1.0];

using Distributions
loss_elbo(d) = -elbo2(d,mm,logsm,γ,δ)
mydata = Iterators.repeated(([3.0;2.0],), 1000)
TH=zeros(4,10);
for epo=1:10
    global plc
    global TH
    Flux.train!(loss_elbo,Flux.params([mm,logsm,γ,δ]),mydata,ADAM(0.01))
    plt(m,s) = pdf(Normal(mm[1],sqrt(exp.(logsm[1]))),m).* pdf.(InverseGamma(γ[1],δ[1]), s)
    contour!(plc,1:0.01:4,0:0.01:0.5,plt,levels=5, linestyle=:dash)
    TH[:,epo].=vcat(mm,logsm, γ, δ);
end
