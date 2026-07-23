import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.lang.reflect.Method;
import java.util.Locale;

public class ExportWaveBasedRoomGolden {
  private static final int[] POINT_IDS = new int[] {122, 121, 53, 35};
  private static final double Z0 = 1.2 * 343.0;
  private static final double TEND = 30.0 / 700.0;

  public static void main(String[] args) throws Exception {
    if (args.length != 1 && args.length != 2) {
      throw new IllegalArgumentException("Usage: ExportWaveBasedRoomGolden input.mph [runstd1]");
    }

    String mphPath = args[0];
    boolean runStd1 =
        args.length == 2 && (args[1].equals("runstd1") || args[1].equals("--run-std1"));

    ModelUtil.initStandalone(true);
    try {
      Model model = ModelUtil.load("wave_based_room", mphPath);
      if (model.sol("sol1").isEmpty()) {
        if (!runStd1) {
          throw new RuntimeException(
              "COMSOL solution sol1 is empty. Re-run with --run-std1 to solve study std1 first.");
        }
        System.out.println("COMSOL sol1 is empty; running study std1 before export.");
        model.study("std1").run();
      }

      double[][] receiverCoordinates =
          ExportWaveBasedRoomReceiverPoints.receiverCoordinatesMeters(model);
      for (int index = 0; index < POINT_IDS.length; ++index) {
        System.out.printf(
            Locale.ROOT,
            "WAVE_GOLDEN_RECEIVER,%d,%.17g,%.17g,%.17g,m%n",
            POINT_IDS[index],
            receiverCoordinates[0][index],
            receiverCoordinates[1][index],
            receiverCoordinates[2][index]);
      }

      double[][] table = readTable(model, "tbl1");
      emitProbeTable(table);
      emitStoredPressureCheck(model);
      System.out.println("Wrote COMSOL wave-based-room golden samples to the batch log.");
    } finally {
      ModelUtil.disconnect();
    }
  }

  private static double[][] readTable(Model model, String tableTag) throws Exception {
    Object table = model.result().table(tableTag);
    for (String methodName : new String[] {"getReal", "getTableData"}) {
      try {
        Method method = table.getClass().getMethod(methodName);
        Object value = method.invoke(table);
        if (value instanceof double[][]) {
          return (double[][]) value;
        }
      } catch (NoSuchMethodException ignored) {
      }
    }
    try {
      Method method = table.getClass().getMethod("getReal", boolean.class);
      Object value = method.invoke(table, false);
      if (value instanceof double[][]) {
        return (double[][]) value;
      }
    } catch (NoSuchMethodException ignored) {
    }
    throw new RuntimeException("Could not read COMSOL table " + tableTag + " as double[][]");
  }

  private static void emitProbeTable(double[][] raw) {
    double[][] rows = normalizeTable(raw);
    if (rows.length < 31) {
      throw new RuntimeException("Expected at least 31 probe table rows, got " + rows.length);
    }
    for (int row = 0; row < rows.length; ++row) {
      double time = rows[row][0];
      if (time < -1.0e-12 || time > TEND + 1.0e-10) {
        continue;
      }
      System.out.printf(
          Locale.ROOT,
          "WAVE_GOLDEN_SAMPLE,%.17g,%.17g,%.17g,%.17g,%.17g%n",
          time,
          rows[row][4] * Z0,
          rows[row][3] * Z0,
          rows[row][2] * Z0,
          rows[row][1] * Z0);
      System.out.printf(
          Locale.ROOT,
          "WAVE_GOLDEN_NORMALIZED,%.17g,%.17g,%.17g,%.17g,%.17g%n",
          time,
          rows[row][4],
          rows[row][3],
          rows[row][2],
          rows[row][1]);
    }
  }

  private static double[][] normalizeTable(double[][] raw) {
    if (raw.length == 0 || raw[0].length == 0) {
      throw new RuntimeException("Probe table is empty.");
    }
    if (raw[0].length == 5) {
      return raw;
    }
    if (raw.length == 5) {
      double[][] transposed = new double[raw[0].length][5];
      for (int row = 0; row < raw[0].length; ++row) {
        for (int col = 0; col < 5; ++col) {
          transposed[row][col] = raw[col][row];
        }
      }
      return transposed;
    }
    throw new RuntimeException(
        "Expected probe table shape Nx5 or 5xN, got " + raw.length + "x" + raw[0].length);
  }

  private static void emitStoredPressureCheck(Model model) throws Exception {
    try {
      model.result().numerical().remove("wavegoldcheck");
    } catch (Exception ignored) {
    }
    model.result().numerical().create("wavegoldcheck", "EvalPoint");
    model.result().numerical("wavegoldcheck").set("data", "dset1");
    model.result().numerical("wavegoldcheck").selection().geom("geom1", 0);
    model.result().numerical("wavegoldcheck").selection().set(POINT_IDS);
    model.result().numerical("wavegoldcheck").setIndex("expr", "pate.p_t", 0);
    model.result().numerical("wavegoldcheck").setIndex("unit", "Pa", 0);
    double[][] pressure = model.result().numerical("wavegoldcheck").getReal(false);
    for (int sample = 0; sample < pressure[0].length; ++sample) {
      double time = sample / 700.0;
      System.out.printf(
          Locale.ROOT,
          "WAVE_GOLDEN_STORED_PA,%.17g,%.17g,%.17g,%.17g,%.17g%n",
          time,
          pressure[0][sample],
          pressure[1][sample],
          pressure[2][sample],
          pressure[3][sample]);
    }
  }
}
