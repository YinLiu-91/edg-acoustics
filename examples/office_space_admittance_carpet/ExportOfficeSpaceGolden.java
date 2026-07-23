import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.util.Locale;

public class ExportOfficeSpaceGolden {
  private static final int[] POINT_IDS = new int[] {230, 233, 467};
  private static final double OUTPUT_DT = (1.0 / 750.0) / 30.0;
  private static final int EXPECTED_NSAMPLES = 9001;

  public static void main(String[] args) throws Exception {
    if (args.length != 1 && args.length != 2) {
      throw new IllegalArgumentException(
          "Usage: ExportOfficeSpaceGolden input.mph [runstd2]");
    }

    String mphPath = args[0];
    boolean runStd2 =
        args.length == 2 && (args[1].equals("runstd2") || args[1].equals("--run-std2"));

    ModelUtil.initStandalone(true);
    try {
      Model model = ModelUtil.load("office_space", mphPath);
      if (model.sol("sol2").isEmpty()) {
        if (!runStd2) {
          throw new RuntimeException(
              "COMSOL solution sol2 is empty. Re-run with --run-std2 to solve study std2 first.");
        }
        System.out.println("COMSOL sol2 is empty; running study std2 before export.");
        model.study("std2").run();
      }

      try {
        model.result().numerical().remove("officegold");
      } catch (Exception ignored) {
      }

      model.result().numerical().create("officegold", "EvalPoint");
      model.result().numerical("officegold").set("data", "dset2");
      model.result().numerical("officegold").selection().geom("geom1", 0);
      model.result().numerical("officegold").selection().set(POINT_IDS);
      model.result().numerical("officegold").setIndex("expr", "pate.p_t", 0);
      model.result().numerical("officegold").setIndex("unit", "Pa", 0);

      double[][] pressure = model.result().numerical("officegold").getReal(false);
      validatePressureShape(pressure);
      double[][] receiverCoordinates =
          ExportOfficeSpaceReceiverPoints.receiverCoordinatesMeters(model);
      writeLog(pressure, receiverCoordinates);
      System.out.println("Wrote COMSOL office-space golden samples to the batch log.");
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

  private static void writeLog(double[][] pressure, double[][] receiverCoordinates) {
    for (int index = 0; index < POINT_IDS.length; ++index) {
      System.out.printf(
          Locale.ROOT,
          "OFFICE_GOLDEN_RECEIVER,%d,%.17g,%.17g,%.17g,m%n",
          POINT_IDS[index],
          receiverCoordinates[0][index],
          receiverCoordinates[1][index],
          receiverCoordinates[2][index]);
    }
    for (int sample = 0; sample < pressure[0].length; ++sample) {
      double time = sample * OUTPUT_DT;
      System.out.printf(
          Locale.ROOT,
          "OFFICE_GOLDEN_SAMPLE,%.17g,%.17g,%.17g,%.17g%n",
          time,
          pressure[0][sample],
          pressure[1][sample],
          pressure[2][sample]);
    }
  }
}
