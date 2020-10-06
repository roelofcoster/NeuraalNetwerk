package NeuraalNetwerk;

import org.ejml.data.DMatrix;
import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;
import java.util.Random;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleFunction;

public class GebruikNetwerk {
	
	public static void main(String[] args) {
		voerUit();
	}
	static private void voerUit(){
		DoubleFunction<Double> f = (double x) -> Functies.elu(x, 1);
		DoubleFunction<Double> f_accent = (double x) -> Functies.elu_accent(x, 1);
//		DoubleFunction<Double> f = (double x) -> Functies.relu(x);
//		DoubleFunction<Double> f_accent = (double x) -> Functies.relu_accent(x);
		DoubleFunction<Double> identiteit = (double x) -> x;
		DoubleFunction<Double> een = (double x) -> 1.0;
		
		DoubleBinaryOperator kost = (double observatie, double voorspelling) -> Math.pow(observatie - voorspelling, 2) / 2;
		DoubleBinaryOperator kost_gradient = (double observatie, double voorspelling) -> voorspelling - observatie;
		
		double[] grenzen_gewichten = new double[]{-.1,.1};
		double[] grenzen_biases = new double[]{0,0};
		
		Netwerk net = new Netwerk(kost, kost_gradient);
		net.maakEersteLaag(1);
		net.maakLaag(10, f, f_accent, grenzen_gewichten, grenzen_biases);
		net.maakLaag(10, f, f_accent, grenzen_gewichten, grenzen_biases);
		net.maakLaag(10, f, f_accent, grenzen_gewichten, grenzen_biases);
		net.maakLaag(10, f, f_accent, grenzen_gewichten, grenzen_biases);
		net.maakLaag(1, identiteit, een, grenzen_gewichten, grenzen_biases);
		
		SimpleMatrix drin, druit;
		
		while(true) {
			drin = SimpleMatrix.random_DDRM((int) 2e4, 1, -1, 1, new Random());
			druit = Functies.sinus(drin, 10);
			if (net.predict(drin).elementMaxAbs() > 0) break;
		}
		double[] fout = new double[100];
		double rate = 1e-5;
		String gebeurtenis = "=";
		
		int maxEpochs = 100;
		int hapgrootte = 32;
		int aantalHappen = drin.numRows() / hapgrootte;
		
		for(int epoch = 0; epoch < maxEpochs; epoch++){
			Functies.permuteer(drin, druit);
			System.out.print(epoch + "\t");
			for(int hap = 0; hap < aantalHappen; hap++) {
				int[] rijen = new int[]{hapgrootte * hap, hapgrootte * (hap + 1) - 1};
				int[] kolommen = new int[]{0, drin.numCols()};
				net.backprop(
								drin.extractMatrix(rijen[0], rijen[1], kolommen[0], kolommen[1]),
								druit.extractMatrix(rijen[0], rijen[1], kolommen[0], kolommen[1]), rate);
			}
			if(hapgrootte * aantalHappen < drin.numRows() - 1){
				int[] rijen = new int[]{hapgrootte * aantalHappen, hapgrootte * drin.numRows() - 1};
				int[] kolommen = new int[]{0, drin.numCols()};
				net.backprop(
								drin.extractMatrix(rijen[0], rijen[1], kolommen[0], kolommen[1]),
								druit.extractMatrix(rijen[0], rijen[1], kolommen[0], kolommen[1]), rate);
			}
			System.out.println(rate +"\t" + gebeurtenis + "\t" + druit.minus(net.predict(drin)).elementMaxAbs() + "\t" + Functies.kost_kwadraat(druit, net.predict(drin)));
			
			int foutTrace = 10;
			gebeurtenis = "=";
			Functies.schuif(fout, net.getKost());
			if(fout[0] == fout[1]) break;
//			if(Functies.opGoedeWeg(fout, foutTrace) && rate < .1){
//					gebeurtenis = "^";
//					rate *= 2;
//			}
//			if(epoch > foutTrace && fout[0] > fout[1]){
//				gebeurtenis = "v";
//				rate /= 10;
//			}
			if(fout[0] < 5e-4) break;
			if(rate < 1e-10) break;
			if(Double.isNaN(fout[0])) break;
		}
		
		SimpleMatrix presentatie = drin.copy();
		presentatie = presentatie.combine(0,1,druit);
		presentatie = presentatie.combine(0,2,net.predict(drin));
		presentatie = presentatie.combine(0,3,druit.minus(net.predict(drin)));

		System.out.println(presentatie.extractMatrix(0,100,0,4));
	}
}
