package NeuraalNetwerk;

import org.ejml.simple.SimpleMatrix;

import java.util.ArrayList;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleFunction;

public class Netwerk {
	private ArrayList<Layer> layers;
	private int nLayers;
	private DoubleBinaryOperator kostfunctie;
	private DoubleBinaryOperator kostgradientfunctie;
	private static int iteraties = 0;
	private double kost;
	
	public Netwerk(DoubleBinaryOperator kostfunctie, DoubleBinaryOperator kostgradientfunctie){
		this.kostfunctie = kostfunctie;
		this.kostgradientfunctie = kostgradientfunctie;
		layers = new ArrayList<Layer>();
		nLayers = 0;
	}
	public void maakEersteLaag(int nNodes){
		layers.add(new Layer(nNodes));
		nLayers++;
	}
	public void maakLaag(int nNodes, DoubleFunction<Double> activering, DoubleFunction<Double> activering_accent, double[] grenzen_gewichten, double[] grenzen_biases){
		layers.add(new Layer(nNodes, layers.get(nLayers - 1).getNNodes(), activering, activering_accent, grenzen_gewichten, grenzen_biases));
		nLayers++;
	}
	
	// Het echte werk
	public SimpleMatrix predict(SimpleMatrix input) throws IllegalArgumentException{
		try {
			layers.get(0).setUitkomst(input);
			if (nLayers > 1) for (int i = 1; i < nLayers; i++) {
				layers.get(i).predict(layers.get(i - 1).getUitkomst());
			}
			return layers.get(nLayers - 1).getUitkomst();
		}
		catch(IllegalArgumentException fout) {
			throw new IllegalArgumentException("Inputdata hebben onjuiste dimensie");
		}
	}
	private void maakGradienten(SimpleMatrix druit){
		iteraties++;
		if(nLayers < 2) throw new RuntimeException("NeuraalNetwerk.Netwerk nog niet opgebouwd");
		int L = nLayers - 1;
		int N = layers.get(0).getUitkomst().numRows();
		SimpleMatrix delta = Functies.kost_gradient(kostgradientfunctie, druit, layers.get(L).getUitkomst());
		SimpleMatrix daux = layers.get(L).getOngeactiveerd();
		Functies.mapFunctie(daux, layers.get(L).getActivering_accent());
		delta = delta.elementMult(daux);
		layers.get(L).setDelta(delta);

		if(nLayers > 2) for(L = nLayers - 2; L > 0; L--){
			delta = layers.get(L + 1).getDelta();
			delta = delta.mult(layers.get(L+1).getGewichten().transpose());
			daux = layers.get(L).getOngeactiveerd();
			Functies.mapFunctie(daux, layers.get(L).getActivering_accent());
			delta = delta.elementMult(daux);
			layers.get(L).setDelta(delta);
		}
		
		for(L = 1; L < nLayers; L++) {
			layers.get(L).setGradient_bias(Functies.kolom_gemiddelde(layers.get(L).getDelta()));
			SimpleMatrix gradient_gewicht = new SimpleMatrix(
							layers.get(L - 1).getNNodes(),
							layers.get(L).getNNodes());
			for (int n = 0; n < N; n++) {
				for (int rij = 0; rij < layers.get(L - 1).getNNodes(); rij++) {
					for (int kolom = 0; kolom < layers.get(L).getNNodes(); kolom++) {
						gradient_gewicht.set(rij, kolom,
										gradient_gewicht.get(rij, kolom) +
													layers.get(L - 1).getUitkomst().get(n, rij) *
													layers.get(L).getDelta().get(n, kolom)
						);
					}
				}
			}
			gradient_gewicht = gradient_gewicht.divide(N);
			layers.get(L).setGradient_gewicht(gradient_gewicht);
		}
	}
	private void berekenKost(SimpleMatrix observatie){
		kost = Functies.kost(kostfunctie, observatie, layers.get(nLayers - 1).getUitkomst());
	}
	public void backprop(SimpleMatrix drin, SimpleMatrix druit, double rate){
		predict(drin);
		berekenKost(druit);
		maakGradienten(druit);
		for(int L = 1; L < nLayers; L++){
			layers.get(L).setGewichten(
							layers.get(L).getGewichten().plus(
							layers.get(L).getGradient_gewicht().divide(1 / rate)));
			layers.get(L).setBiases(
							layers.get(L).getBiases().plus(
							layers.get(L).getGradient_bias().divide(1 / rate)));
		}
	}
	
	// getters
	public SimpleMatrix getUitkomstLaag(int n){
		if(n > layers.size() + 1) throw new IllegalArgumentException("Laag " + n + "bestaat niet; netwerk heeft " + nLayers + " lagen");
		return layers.get(n).getUitkomst();
	}
	public SimpleMatrix getGewichten(int layer){
		try{
			return layers.get(layer).getGewichten();
		} catch (IndexOutOfBoundsException e) {
			System.out.println("NeuraalNetwerk.Netwerk heeft geen " + layer + " lagen");
		}
		return null;
	}
	public SimpleMatrix getUitkomst(int layer){
		try{
			return layers.get(layer).getUitkomst();
		} catch (IndexOutOfBoundsException e) {
			System.out.println("NeuraalNetwerk.Netwerk heeft geen " + layer + " lagen");
		}
		return null;
	}
	public int getNLayers(){
		return nLayers;
	}
	public int[] getVorm(){
		if(nLayers == 0) return null;
		int[] uitk = new int[nLayers];
		for(int i = 0; i < nLayers; i++)
			uitk[i]=layers.get(i).getNNodes();
		return uitk;
	}
	public Layer getLaag(int n){
		return layers.get(n);
	}
	public DoubleBinaryOperator getKostfunctie(){
		return kostfunctie;
	}
	public DoubleBinaryOperator getKostgradientfunctie(){
		return kostgradientfunctie;
	}
	public double getKost(){
		return kost;
	}
}
