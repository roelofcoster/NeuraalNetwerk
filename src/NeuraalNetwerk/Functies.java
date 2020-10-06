package NeuraalNetwerk;

import org.ejml.data.DMatrixRMaj;
import org.ejml.simple.SimpleMatrix;
import org.jetbrains.annotations.NotNull;
import java.util.ArrayList;
import java.util.Collections;
import java.util.InputMismatchException;
import java.util.function.DoubleBinaryOperator;
import java.util.function.DoubleFunction;

public final class Functies {
	private Functies(){
		throw new IllegalAccessError("Functies niet bedoeld als object");
	}
	public static double relu(double x){
		return Math.max(0, x);
	}
	public static void relu(@NotNull SimpleMatrix x){
		for(int i = 0; i < x.numRows(); i++) for(int j = 0; j < x.numCols(); j++){
			x.set(i,j,relu(x.get(i,j)));
		}
	}
	public static double relu_accent(double x){
		return x > 0?1:0;
	}
	public static void relu_accent(@NotNull SimpleMatrix x){
		for(int i = 0; i < x.numRows(); i++) for(int j = 0; j < x.numCols(); j++){
			x.set(i,j,relu_accent(x.get(i,j)));
		}
	}
	public static double elu(double x, double alfa){
		double y = x;
		if(x < 0) y = alfa * (Math.exp(x) - 1);
		return y;
	}
	public static double elu_accent(double x, double alfa){
		double y;
		y = x > 0 ? 1 : elu(x, alfa) + alfa;
		return y;
	}
	public static void elu(SimpleMatrix x, double alfa){
		for(int i = 0; i < x.numRows(); i++) for(int j = 0; j < x.numCols(); j++){
			x.set(i,j,elu(x.get(i,j), alfa));
		}
	}
	public static void elu_accent(SimpleMatrix x, double alfa){
		for(int i = 0; i < x.numRows(); i++) for(int j = 0; j < x.numCols(); j++){
			x.set(i,j,elu_accent(x.get(i,j), alfa));
		}
	}
	public static void kolomsgewijs_optellen(SimpleMatrix x, SimpleMatrix b){
		if(b.numCols() != 1 || b.numRows() != x.numRows()) throw new InputMismatchException("Dimensies op te tellen matrices niet goed");
		for(int kolom = 0; kolom < x.numCols(); kolom++) for(int rij = 0; rij < x.numRows(); rij++)
			x.set(rij,kolom, x.get(rij,kolom) + b.get(rij,0));
	}
	public static void rijgewijs_optellen(SimpleMatrix x, SimpleMatrix b) {
		if (b.numRows() != 1 || b.numCols() != x.numCols())
			throw new InputMismatchException("Dimensies op te tellen matrices niet goed");
		for (int rij = 0; rij < x.numRows(); rij++)
			for (int kolom = 0; kolom < x.numCols(); kolom++)
				x.set(rij, kolom, x.get(rij, kolom) + b.get(0, kolom));
	}
	public static double kost_kwadraat(SimpleMatrix uitk, SimpleMatrix doel){
		double resultaat = 0;
		for(int rij = 0; rij < uitk.numRows(); rij++) for(int kolom = 0; kolom < uitk.numCols(); kolom++)
			resultaat += Math.pow(uitk.get(rij, kolom) - doel.get(rij, kolom), 2);
		resultaat = resultaat / uitk.numRows() / 2;
		return resultaat;
	}
	public static SimpleMatrix kost_kwadraat_gradient(SimpleMatrix observatie, SimpleMatrix voorspelling){
		return observatie.minus(voorspelling);
	}
	public static SimpleMatrix kolom_gemiddelde(@NotNull SimpleMatrix drin){
		SimpleMatrix druit = new SimpleMatrix(1, drin.numCols());
		for(int kolom = 0; kolom < drin.numCols(); kolom++){
			druit.set(0,kolom, drin.get(0, kolom));
			for(int rij = 1; rij < drin.numRows(); rij++)
				druit.set(0, kolom, druit.get(0, kolom) + drin.get(rij, kolom));
		}
		druit = druit.divide(drin.numRows());
//		for(int kolom = 0; kolom < drin.numCols(); kolom++)
//			druit.set(0, kolom, druit.get(0, kolom) / drin.numRows());
		return druit;
	}
	public static boolean isNAN(SimpleMatrix b){
		for(int rij = 0; rij < b.numRows(); rij++) for(int kolom = 0; kolom < b.numCols(); kolom++){
			if(Double.isNaN(b.get(rij, kolom))) return true;
		}
		return false;
	}
	public static boolean isInfinite(SimpleMatrix b){
		for(int rij = 0; rij < b.numRows(); rij++) for(int kolom = 0; kolom < b.numCols(); kolom++){
			if(Double.isInfinite(b.get(rij, kolom))) return true;
		}
		return false;
	}
	public static SimpleMatrix sinus(SimpleMatrix x, double lambda){
		SimpleMatrix uitk = new SimpleMatrix(x.numRows(), x.numCols());
		for(int rij = 0; rij < x.numRows(); rij++) for(int kolom = 0; kolom < x.numCols(); kolom++){
			uitk.set(rij, kolom, Math.sin(lambda * x.get(rij, kolom)));
		}
		return uitk;
	}
	public static void schuif(double[] x, double nieuw) {
		for (int i = x.length - 1; i > 0; i--) {
			x[i] = x[i - 1];
		}
		x[0] = nieuw;
		}
	private static ArrayList<Integer> indices = new ArrayList<Integer>();
	public static void permuteer(SimpleMatrix x, SimpleMatrix y){
		if(indices.isEmpty()) for(int i = 0; i < x.numRows(); i++){
			indices.add(i);
		}
		Collections.shuffle(indices);
		SimpleMatrix x2 = new SimpleMatrix(x.numRows(), x.numCols());
		SimpleMatrix y2 = new SimpleMatrix(x.numRows(), x.numCols());
		for(int i = 0; i < x.numRows(); i++){
			x2.setRow(i, 0, ((DMatrixRMaj)x.extractVector(true, indices.get(i)).getMatrix()).getData());
			y2.setRow(i, 0, ((DMatrixRMaj)y.extractVector(true, indices.get(i)).getMatrix()).getData());
		}
		x = x2;
		y = y2;
	 }
	public static boolean opGoedeWeg(double[] fouten, int aantal){
		boolean uitk = true;
		for(int i = 0; i < Math.min(fouten.length - 1, aantal); i++){
			uitk = uitk && fouten[i] < fouten[i + 1];
			if(!uitk) break;
		}
		return uitk;
	}
	public static void mapFunctie(SimpleMatrix x, DoubleFunction<Double> f){
		for(int rij = 0; rij < x.numRows(); rij++) for(int kolom = 0; kolom < x.numCols(); kolom++){
			x.set(rij, kolom, f.apply(x.get(rij, kolom)));
		}
	}
	public static double mapKostfunctie(SimpleMatrix x, SimpleMatrix y, DoubleBinaryOperator f){
		double uitkomst = 0;
		for(int rij = 0; rij < x.numRows(); rij++) for(int kolom = 0; kolom < x.numCols(); kolom++){
			uitkomst += f.applyAsDouble(x.get(rij, kolom), y.get(rij, kolom));
		}
		uitkomst /= (x.numRows() * x.numCols());
		return uitkomst;
	}
	public static double kost(DoubleBinaryOperator f, SimpleMatrix observatie, SimpleMatrix voorspelling){
		double resultaat = 0;
		for(int rij = 0; rij < voorspelling.numRows(); rij++) for(int kolom = 0; kolom < voorspelling.numCols(); kolom++)
			resultaat += Math.pow(voorspelling.get(rij, kolom) - observatie.get(rij, kolom), 2);
		resultaat = resultaat / voorspelling.numRows() / 2;
		return resultaat;
	}
	public static SimpleMatrix kost_gradient(DoubleBinaryOperator f, SimpleMatrix observatie, SimpleMatrix voorspelling){
		SimpleMatrix uitkomst = new SimpleMatrix(observatie.numRows(), observatie.numCols());
		for(int rij = 0; rij < observatie.numRows(); rij++) for(int kolom = 0; kolom < observatie.numCols(); kolom++){
			uitkomst.set(rij, kolom,
					f.applyAsDouble(voorspelling.get(rij, kolom), observatie.get(rij, kolom)));
		}
		return uitkomst;
	}
}
