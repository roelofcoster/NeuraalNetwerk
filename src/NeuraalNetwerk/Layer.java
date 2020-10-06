package NeuraalNetwerk;

import org.ejml.simple.SimpleMatrix;
import java.util.Random;
import java.util.function.DoubleFunction;

public class Layer {
	private int nNodes;
	private SimpleMatrix gewichten;
	private SimpleMatrix biases;
	private SimpleMatrix ongeactiveerd;
	private SimpleMatrix uitkomst;
	private SimpleMatrix delta;
	private SimpleMatrix gradient_gewicht;
	private SimpleMatrix gradient_bias;
	private DoubleFunction<Double> activering;
	private DoubleFunction<Double> activering_accent;
	
	public Layer(int nNodes, int nNodes_vorige, DoubleFunction<Double> activering,
							 DoubleFunction<Double> activering_accent, double[] grenzen_gewichten, double[] grenzen_biases) {
		this.nNodes = nNodes;
		gewichten = SimpleMatrix.random_DDRM(nNodes_vorige, nNodes, grenzen_gewichten[0], grenzen_gewichten[1], new Random());
		biases = SimpleMatrix.random_DDRM(1, nNodes, grenzen_biases[0], grenzen_biases[1], new Random());
		this.activering = activering;
		this.activering_accent = activering_accent;
	}
	public Layer(int nNodes) {
		this.nNodes = nNodes;
	}
	
	// Het echte werk
	public void predict(SimpleMatrix input) {
		ongeactiveerd = input.mult(gewichten).copy();
		Functies.rijgewijs_optellen(ongeactiveerd, biases);
		uitkomst = ongeactiveerd.copy();
		Functies.mapFunctie(uitkomst, activering);
	}
	
	// Setters
	public void setUitkomst(SimpleMatrix uitkomst) throws IllegalArgumentException {
		if(uitkomst.numCols() != nNodes)
			throw new IllegalArgumentException("Onjuiste dimensie van uitkomstmatrix");
		this.uitkomst = uitkomst.copy();
	}
	public void setGradient_gewicht(SimpleMatrix gradient_gewicht) {
		this.gradient_gewicht = gradient_gewicht.copy();
	}
	public void setGradient_bias(SimpleMatrix gradient_bias) {
		this.gradient_bias = gradient_bias.copy();
	}
	public void setDelta(SimpleMatrix delta){
		if(delta.numCols() != nNodes)
			throw new IllegalArgumentException("Onjuiste dimensie van uitkomstmatrix");
		this.delta = delta.copy();
	}
	public void setGewichten(SimpleMatrix gw){
		this.gewichten = gw.copy();
	}
	public void setBiases(SimpleMatrix bs){
		this.biases = bs.copy();
	}
	// Getters
	public SimpleMatrix getUitkomst(){
		return uitkomst.copy();
	}
	public int getNNodes() {
		return nNodes;
	}
	public SimpleMatrix getGewichten(){
		return gewichten.copy();
	}
	public SimpleMatrix getBiases(){
		return biases.copy();
	}
	public SimpleMatrix getDelta() {
		return delta.copy();
	}
	public SimpleMatrix getOngeactiveerd() {
		return ongeactiveerd.copy();
	}
	public SimpleMatrix getGradient_gewicht() {
		return gradient_gewicht;
	}
	public SimpleMatrix getGradient_bias() {
		return gradient_bias;
	}
	public DoubleFunction<Double> getActivering(){
		return activering;
	}
	public DoubleFunction<Double> getActivering_accent(){
		return activering_accent;
	}
}
