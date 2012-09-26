import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.*;


public class SAOpt {
	
//	static String citiPath = "./test2.txt";
//	static int totalTime = 3000;
//	static String citiPath = "./miniTest";
	static String citiPath = "./tsp_input.txt";
	static int totalTime = 115000;
	static int NUM_THREAD = 8;
	static int range = 4;
	
	static int n; // Number of cities
	static int m = 60; // Maximum swapping cities
	static int MAXNUM = 1000; // Maximal number of cities
	static int[][] cities;
	static float[][] cityMatrix;
	int id;
	State optimum;
	HighQualityRandom r = new HighQualityRandom();
//	Random r = new Random();	
	static int neighbors[][];
	
	static class State {
		int[] citySequence;
		int[] reverseSequence;
		double fit;
		
		public State() {}
		public State(State y) {
			this.citySequence = new int[n];
			this.reverseSequence = new int[n];
			for (int i = 0; i < n; ++ i) {
				this.citySequence[i] = y.citySequence[i];
				this.reverseSequence[i] = y.reverseSequence[i];
			}
			this.fit = y.fit;
		}
		public State(int N) {
			this.citySequence = new int[N];
			this.reverseSequence = new int[N];
		}
		public void resetReverseSequence() {
			for (int i = 0; i < n; ++i) {
				reverseSequence[citySequence[i]] = i;
			}
		}
	}
	
	public static void calDistance() {
		int i, j;
		for (i = 0; i < cities.length; ++ i) {
			for (j = i + 1; j < cities.length; ++ j) {
				cityMatrix[i][j] = (float)Math.sqrt(Math.pow(cities[i][0]-cities[j][0], 2) + Math.pow(cities[i][1]-cities[j][1], 2) + Math.pow(cities[i][2]-cities[j][2], 2));
				cityMatrix[j][i] = cityMatrix[i][j];
			}
		}
	}
	
	public static double calFitness (State s) {
		double sum = cityMatrix[s.citySequence[0]][s.citySequence[n-1]];
		for (int i = 1; i < n; ++ i) {
			sum += cityMatrix[s.citySequence[i]][s.citySequence[i-1]];
		}
		return sum;
	}
	
	public double calFitness (int p_start, int q_start, int p_end, int q_end, State y) {
		int p_before_start = p_start - 1 + n; if (p_before_start >= n) p_before_start -= n;
		int q_before_start = q_start - 1 + n; if (q_before_start >= n) q_before_start -= n;
		int p_before_end = p_end - 1 + n; if (p_before_end >= n) p_before_end -= n;
		int q_before_end = q_end - 1 + n; if (q_before_end >= n) q_before_end -= n;
		float diff = cityMatrix[y.citySequence[p_before_start]][y.citySequence[q_start]] + cityMatrix[y.citySequence[q_before_end]][y.citySequence[p_end]]
				+ cityMatrix[y.citySequence[q_before_start]][y.citySequence[p_start]] + cityMatrix[y.citySequence[p_before_end]][y.citySequence[q_end]]
				- cityMatrix[y.citySequence[p_before_start]][y.citySequence[p_start]] - cityMatrix[y.citySequence[p_before_end]][y.citySequence[p_end]]
				- cityMatrix[y.citySequence[q_before_start]][y.citySequence[q_start]] - cityMatrix[y.citySequence[q_before_end]][y.citySequence[q_end]];
		return y.fit + diff;
	}
	
	public void iterate2(State s, State yy) {
		long t = r.nextLong();
		int x = (int)(t & 0xFFFF) % n;
		int w = (int)((t >> 16) & 0xFFFF);
		int b = 0xFFFF;
		int a = (2 * 0xFFFF) * range;
		int p = (int)(a / (w + b) - range);
//		int y = s.reverseSequence[neighbors[s.citySequence[x]][(int)((t >> 16) & 0xFFFF) % 4 + 1]];
		int y = s.reverseSequence[neighbors[s.citySequence[x]][p + 1]];
		if (x > y) {
			int q = x; x = y; y = q;
		}
		if (y == (x + 1) || y == (n - 1)) {
			return;
		}
		float cost = cityMatrix[s.citySequence[x]][s.citySequence[y]] + cityMatrix[s.citySequence[x + 1]][s.citySequence[y + 1]]
				- cityMatrix[s.citySequence[x]][s.citySequence[x + 1]] - cityMatrix[s.citySequence[y]][s.citySequence[y + 1]];
		if (cost >= 0 && (t >> 55) != 0) return;
		for (int i = 0; i < y - x; ++i) {
			yy.citySequence[i] = s.citySequence[y - i];
		}
		for (int i = 0; i < n - (y + 1); ++i) {
			yy.citySequence[y - x + i] = s.citySequence[i + (y + 1)];
		}
		for (int i = 0; i <= x; ++i) {
			yy.citySequence[n - x - 1 + i] = s.citySequence[i];
		}
		yy.fit = s.fit + cost;
		double compare = calFitness(yy);
		if (Math.abs(compare - yy.fit) > 1) throw new RuntimeException();
		System.arraycopy(yy.citySequence, 0, s.citySequence, 0, s.citySequence.length);
		s.fit = yy.fit;
		s.resetReverseSequence();
	}
	
	public void iterate (State s, State y) {
		
		long x = r.nextLong();
		int x1 = (int)(x & 0xFFFF);
		int x2 = (int)((x >> 16) & 0xFFFF);
		int x3 = (int)((x >> 32) & 0xFFFF);
		int x4 = (int)((x >> 48) & 0xFFFF);

		int a, b, c, d, p_start, q_start, p_end, q_end;
		a = x1 % n;
		b = 1 + x2 % Math.min(n / 2, m);
		c = 1 + x3 % (n - b - 2);
		d = 1 + x4 % (Math.min(n - b - c - 1, m));
		
		p_start = a; if (p_start >= n) p_start -= n;
		p_end = (a + b) % n; if (p_end >= n) p_end -= n;
		q_start = (a + b + c) % n; if (q_start >= n) q_start -= n;
		q_end = (a + b + c + d) % n; if (q_end >= n) q_end -= n;
		
		y.fit = calFitness(p_start, q_start, p_end, q_end, s);
		
//		renewSeq(temperature, a, b, c, d, p_start, q_start, p_end, q_end, s, y);
		double cost = y.fit - s.fit;
		if (cost < 0) {
			for (int i = 0; i < d; ++i) {
				y.citySequence[i] = s.citySequence[(q_start + i) % n];
			}
			for (int i = 0; i < c; ++i) {
				y.citySequence[d + i] = s.citySequence[(p_end + i) % n];
			}
			for (int i = 0; i < b; ++i) {
				y.citySequence[d + c + i] = s.citySequence[(p_start + i) % n];
			}
			for (int i = 0; i < n - b - c - d; ++i) {
				y.citySequence[d + c + b + i] = s.citySequence[(q_end + i) % n];
			}
//			double compare = calFitness(y);
//			if (Math.abs(compare - y.fit) > 0.0001) throw new RuntimeException();
			System.arraycopy(y.citySequence, 0, s.citySequence, 0, s.citySequence.length);
			s.fit = y.fit;
			s.resetReverseSequence();
		}
	}

	
	public int[][] readCities (String citiPath) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(citiPath)));
		String line = "";
		int[][] rawData = new int[MAXNUM][3];
		int j = 0, i;
		String[] di;
		
		while ((line = br.readLine()) != null) {
			di = line.split(" ");
			for (i = 1; i < di.length; ++ i) {
				rawData[j][i - 1] = Integer.parseInt(di[i]);
			}
//			System.out.println(rawData[j][0] + " " + rawData[j][1] + " " + rawData[j][2]);
			++ j;
		}
		n = j;
		br.close();
		return rawData;
	}
	
	public void SAModel (long start_time, State xxx) {
		

		
		// Sample state space with 1% of runs to determine temperature schedule
		State y = new State(n);
		
		// Simulated Annealing
		State x = new State(xxx);
//		optimum.fit = x.fit;
		optimum = new State(x);
		
		do {
//			iterate(x, y);
			iterate2(x, y);
			if (x.fit < optimum.fit) {
				System.out.println("[" + (System.currentTimeMillis() - start_time) / 1000 + "][" + id + "]Better Score: " + x.fit);
				optimum = new State(x);
			}
		} while(System.currentTimeMillis() - start_time < totalTime);
		
		// Reheat if stagnant
//		if (l - optgen <= STAGFACTOR * ITS)
//			Dtemp = Math.pow(Dtemp, STAGFACTOR * l / ITS);
		
//		System.out.println("Iteration num: " + l);
//		System.out.println("Optgen num:" + optgen);
//		System.out.println("Acceptance num: " + acceptance);
	}
	
	public SAOpt(int id) {
		int N = n;
		this.id = id;
		this.optimum = new State(N);
	}
		
	public static void main(String[] args) throws IOException {
		final long start_time = System.currentTimeMillis(); // Calculating running time. Stop when reaches 2 min
		int tmp, i, j, begin = 0;
		
		//		String citiPath = "./miniTest";

		BufferedReader br = new BufferedReader(new FileReader(new File(citiPath)));
		String line = "";
		int[][] rawData = new int[MAXNUM][3];
		int jj = 0, ii;
		String[] di;
		
		while ((line = br.readLine()) != null) {
			di = line.split(" ");
			for (ii = 1; ii < di.length; ++ ii) {
				rawData[jj][ii - 1] = Integer.parseInt(di[ii]);
			}
//			System.out.println(rawData[j][0] + " " + rawData[j][1] + " " + rawData[j][2]);
			++ jj;
		}
		n = jj;
		br.close();
				
		cities = rawData;
		SAOpt.cityMatrix = new float[n][n];
		calDistance();
		final State x = new State(n);
		
		///////////
		
		MST mst = new MST(cityMatrix);
		System.out.println("MST: " + mst.getSum());
		mst.fillSequence(x.citySequence);
		x.resetReverseSequence();
		x.fit = calFitness(x);
		System.out.println("Init: " + x.fit);
		State xx = new State(x);
		
		
		neighbors = new int[n][n];
		Vector<Integer> sortNeighbors = new Vector<Integer>();
		for (int u = 0; u < n; ++u) {
			final int uu = u;
			sortNeighbors.clear();
			for (int v = 0; v < n; ++v) {
				sortNeighbors.add(v);
			}
			Collections.sort(sortNeighbors, new Comparator<Integer>() {
				@Override
				public int compare(Integer o1, Integer o2) {
					float x = SAOpt.cityMatrix[uu][o1];
					float y = SAOpt.cityMatrix[uu][o2];
					if (x < y) return -1;
					if (x > y) return 1;
					return 0;
				}
			});
			for (int v = 0; v < n; ++v) {
				neighbors[u][v] = sortNeighbors.get(v);
			}
		}
		
		///////////

		Vector<Thread> threads = new Vector<Thread>();
		final Vector<SAOpt> saopts = new Vector<SAOpt>();
		for (int t = 0; t < NUM_THREAD; ++t) {
			saopts.add(new SAOpt(t));
		}
		
		for (int t = 0; t < NUM_THREAD; ++t) {
			final int id = t;
			Thread th = new Thread() {
				public void run() {
					SAOpt sa = saopts.get(id);
					sa.SAModel(start_time, x);
				}
			};
			th.start();
			threads.add(th);
		}
		for (int t = 0; t < NUM_THREAD; ++t) {
			try {
				threads.get(t).join();
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		
		int min_saopt = -1;
		double min_fit = 999999999;
		for (int t = 0; t < NUM_THREAD; ++t) {
			if (saopts.get(t).optimum.fit < min_fit) {
				min_saopt = t;
				min_fit = saopts.get(t).optimum.fit;
			}
		}
		
		SAOpt sa = saopts.get(min_saopt);
		
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("./tsp_output.txt")));
		for (i = 0; i < sa.n; ++ i) {
			if (sa.optimum.citySequence[i] == 0) {
				begin = i;
				break;
			}
		}
		for (i = begin; i < sa.n; ++ i) {
			tmp = sa.optimum.citySequence[i] + 1;
			System.out.println(tmp);
			bw.write(tmp + "\r\n");
		}
		for (i = 0; i < begin; ++ i) {
			tmp = sa.optimum.citySequence[i] + 1;
			System.out.println(tmp);
			bw.write(tmp + "\r\n");
		}
		tmp = sa.optimum.citySequence[begin] + 1;
		bw.write(tmp + "\r\n");
		System.out.println(tmp + "\r\n");
		bw.close();
		System.out.println("Score: " + sa.optimum.fit);
		long end_time = System.currentTimeMillis();
		long duration = end_time - start_time;
		System.out.println("Time used: " + duration);
		
	}
	
	/////////////// Calculating all the possible scores by permutation of city sequences ////////////////
	public void testSwap (int[] in) {
		int a, b, c, d, p_start, p_end, q_start, q_end;
		int[] out = new int[n];
		Random r = new Random();
		a = r.nextInt(n);
		b = 1 + r.nextInt(n / 2);
		c = r.nextInt(n - b - 1);
		d = 1 + r.nextInt(n - b - c - 1);

		p_start = a;
		p_end = (a + b) % n;
		q_start = (a + b + c) % n;
		q_end = (a + b + c + d) % n;
		
		for (int i = 0; i < d; ++i) {
			out[i] = in[(q_start + i) % n];
		}
		for (int i = 0; i < c; ++i) {
			out[d + i] = in[(p_end + i) % n];
		}
		for (int i = 0; i < b; ++i) {
			out[d + c + i] = in[(p_start + i) % n];
		}
		for (int i = 0; i < n - b - c - d; ++i) {
			out[d + c + b + i] = in[(q_end + i) % n];
		}
		System.out.println();
	}
	
	public void findOptimal(State sequence) {
		
		findOptimal(1, sequence);
	}
	double globalFit = Integer.MAX_VALUE;
	public void findOptimal (int index, State sequence) {
		if (index == sequence.citySequence.length) {
			/*for (int j = 0; j < sequence.citiSequence.length; ++ j) {
				System.out.print(sequence.citiSequence[j]);
			}
			System.out.println();*/
			sequence.fit = calFitness(sequence);
			if (sequence.fit < globalFit) {
				globalFit = sequence.fit;
				optimum = new State(sequence);
//				System.out.println(globalFit);
			}
			return;
		}
		else {
			for (int i = index; i <= sequence.citySequence.length; ++ i) {
				swap (sequence.citySequence, index, i - 1);
				findOptimal(index + 1, sequence);
				swap(sequence.citySequence, index, i - 1);
			}
		}
	}
	public void swap (int[] sequence, int index, int i) {
		int c = sequence[index];
		sequence[index] = sequence[i];
		sequence[i] = c;
	}
	
}


@SuppressWarnings("serial")
class HighQualityRandom extends Random {
	  private long u;
	  private long v = 4101842887655102017L;
	  private long w = 1;
	  
	  public HighQualityRandom() {
	    this(System.nanoTime());
	  }
	  public HighQualityRandom(long seed) {
	    u = seed ^ v;
	    nextLong();
	    v = u;
	    nextLong();
	    w = v;
	    nextLong();
	  }
	  
	  public long nextLong() {
	      u = u * 2862933555777941757L + 7046029254386353087L;
	      v ^= v >>> 17;
	      v ^= v << 31;
	      v ^= v >>> 8;
	      w = 4294957665L * (w & 0xffffffff) + (w >>> 32);
	      long x = u ^ (u << 21);
	      x ^= x >>> 35;
	      x ^= x << 4;
	      long ret = (x + v) ^ w;
	      return ret;
	  }
	  
	  protected int next(int bits) {
	    return (int) (nextLong() >>> (64-bits));
	  }

	}

