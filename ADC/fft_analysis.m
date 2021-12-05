Fs = 50*10^3;
L = 1024;

S1 = csvread("adc_data_irq_1khz_numerical.txt");
S2 = csvread("adc_data_dma_1khz_numerical.txt");

X1 = fft(S1,1024);
P21 = abs(X1/L);
P11 = P21(1:L/2+1);
P11(2:end-1) = 2*P11(2:end-1);

f = Fs*(0:L/2)/L;

figure
plot(f,P11)
title('Single-Sided Amplitude Spectrum of S1')
xlim([0,5000]);
xlabel('f (Hz)')
ylabel('|P1(f)|')


X2 = fft(S2,1024);
P22 = abs(X2/L);
P12 = P22(1:L/2+1);
P12(2:end-1) = 2*P12(2:end-1);

f = Fs*(0:L/2)/L;

figure
plot(f,P12)
title('Single-Sided Amplitude Spectrum of S2')
xlim([0,5000]);
xlabel('f (Hz)')
ylabel('|P1(f)|')


figure
sinad(S1, 50000);
figure
sinad(S2,50000);

sinad_1 = 54.14;
sinad_2 = 59.50;

enob_1 = (sinad_1 + 20*log10(4095/1313) - 1.76 ) / 6.02
enob_2 = (sinad_2 + 20*log10(4095/1313) - 1.76 ) / 6.02