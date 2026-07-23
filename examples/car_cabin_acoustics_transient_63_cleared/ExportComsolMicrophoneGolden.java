import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.PrintWriter;
import java.util.Locale;

public class ExportComsolMicrophoneGolden {
  private static final int[] POINT_IDS = new int[] {197, 391, 402};
  private static final double[][] RECEIVER_COORDS =
      new double[][] {
        {2.0, 2.5, 2.5},
        {-0.05, -0.55, 0.55},
        {1.2, 1.2, 1.2}
      };
  private static final double OUTPUT_DT = 0.001 / 40.0;
  private static final int EXPECTED_NSAMPLES = 2401;

  public static void main(String[] args) throws Exception {
    if (args.length != 2) {
      throw new IllegalArgumentException(
          "Usage: ExportComsolMicrophoneGolden input.mph output.csv");
    }

    String mphPath = args[0];
    String csvPath = args[1];

    ModelUtil.initStandalone(true);
    try {
      Model model = ModelUtil.load("car_cabin", mphPath);
      if (model.sol("sol2").isEmpty()) {
        throw new RuntimeException(
            "COMSOL solution sol2 is empty. Run study std2 first, then export the golden file.");
      }

      try {
        model.result().numerical().remove("micgold");
      } catch (Exception ignored) {
      }

      model.result().numerical().create("micgold", "EvalPoint");
      model.result().numerical("micgold").set("data", "dset2");
      model.result().numerical("micgold").selection().geom("geom1", 0);
      model.result().numerical("micgold").selection().set(POINT_IDS);
      model.result().numerical("micgold").setIndex("expr", "pate.p_t", 0);
      model.result().numerical("micgold").setIndex("unit", "Pa", 0);

      double[][] pressure = model.result().numerical("micgold").getReal(false);
      validatePressureShape(pressure);
      writeCsv(csvPath, pressure);
      System.out.println("Wrote COMSOL microphone golden: " + csvPath);
    } finally {
      ModelUtil.disconnect();
    }
  }

  private static void validatePressureShape(double[][] pressure) {
    if (pressure.length != POINT_IDS.length) {
      throw new RuntimeException(
          "Expected "
              + POINT_IDS.length
              + " receiver rows from EvalPoint, got "
              + pressure.length);
    }
    for (int i = 0; i < pressure.length; ++i) {
      if (pressure[i].length != EXPECTED_NSAMPLES) {
        throw new RuntimeException(
            "Expected "
                + EXPECTED_NSAMPLES
                + " time samples for point "
                + POINT_IDS[i]
                + ", got "
                + pressure[i].length
                + ". Check dset2 output times before using this file as golden.");
      }
    }
  }

  private static void writeCsv(String csvPath, double[][] pressure) throws Exception {
    try (PrintWriter out = new PrintWriter(csvPath)) {
      out.println("# COMSOL Microphone Response pg12/ptgr1");
      out.println("# dataset,dset2");
      out.println("# expression,pate.p_t");
      out.println("# unit,Pa");
      out.println("# receiver_point_ids,197,391,402");
      out.println("# receiver_coords_x,2.0,2.5,2.5");
      out.println("# receiver_coords_y,-0.05,-0.55,0.55");
      out.println("# receiver_coords_z,1.2,1.2,1.2");
      out.println("time,p197,p391,p402");
      for (int sample = 0; sample < pressure[0].length; ++sample) {
        double time = sample * OUTPUT_DT;
        out.printf(
            Locale.ROOT,
            "%.17g,%.17g,%.17g,%.17g%n",
            time,
            pressure[0][sample],
            pressure[1][sample],
            pressure[2][sample]);
      }
    }
  }
}
