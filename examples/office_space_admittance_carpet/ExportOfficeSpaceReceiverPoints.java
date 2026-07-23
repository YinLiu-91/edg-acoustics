import com.comsol.model.GeomSequence;
import com.comsol.model.Model;
import com.comsol.model.util.ModelUtil;
import java.io.PrintWriter;
import java.util.Locale;

public class ExportOfficeSpaceReceiverPoints {
  private static final int[] POINT_IDS = new int[] {230, 233, 467};

  public static void main(String[] args) throws Exception {
    if (args.length != 1 && args.length != 2) {
      throw new IllegalArgumentException(
          "Usage: ExportOfficeSpaceReceiverPoints input.mph [output.json]");
    }

    ModelUtil.initStandalone(true);
    try {
      Model model = ModelUtil.load("office_space_receiver_points", args[0]);
      GeomSequence geometry = model.component("comp1").geom("geom1");
      String lengthUnit = geometry.lengthUnit();
      double scaleToMeters = scaleToMeters(lengthUnit);
      double[][] vertices = geometry.getVertexCoord();
      validateVertices(vertices);
      double[][] receiver = receiverCoordinatesMeters(vertices, scaleToMeters);
      for (int index = 0; index < POINT_IDS.length; ++index) {
        System.out.printf(
            Locale.ROOT,
            "OFFICE_RECEIVER_COORDINATE,%d,%.17g,%.17g,%.17g,m%n",
            POINT_IDS[index],
            receiver[0][index],
            receiver[1][index],
            receiver[2][index]);
      }
      if (args.length == 2) {
        writeJson(args[1], lengthUnit, receiver);
        System.out.println(
            "Wrote COMSOL receiver coordinates for point entities 230, 233, 467: " + args[1]);
      }
    } finally {
      ModelUtil.disconnect();
    }
  }

  static double[][] receiverCoordinatesMeters(Model model) {
    GeomSequence geometry = model.component("comp1").geom("geom1");
    double scale = scaleToMeters(geometry.lengthUnit());
    double[][] vertices = geometry.getVertexCoord();
    validateVertices(vertices);
    return receiverCoordinatesMeters(vertices, scale);
  }

  private static double[][] receiverCoordinatesMeters(double[][] vertices, double scale) {
    double[][] receiver = new double[3][POINT_IDS.length];
    for (int dimension = 0; dimension < 3; ++dimension) {
      for (int index = 0; index < POINT_IDS.length; ++index) {
        receiver[dimension][index] = vertices[dimension][POINT_IDS[index] - 1] * scale;
      }
    }
    return receiver;
  }

  private static void writeJson(String path, String lengthUnit, double[][] receiver)
      throws Exception {
    try (PrintWriter out = new PrintWriter(path)) {
      out.println("{");
      out.println("  \"point_ids\": [230, 233, 467],");
      out.println("  \"coords\": [");
      for (int dimension = 0; dimension < 3; ++dimension) {
        out.print("    [");
        for (int index = 0; index < POINT_IDS.length; ++index) {
          if (index > 0) {
            out.print(", ");
          }
          out.printf(Locale.ROOT, "%.17g", receiver[dimension][index]);
        }
        out.println(dimension == 2 ? "]" : "],");
      }
      out.println("  ],");
      out.printf(Locale.ROOT, "  \"geometry_length_unit\": \"%s\",%n", lengthUnit);
      out.println("  \"coordinate_unit\": \"m\",");
      out.println("  \"source\": \"COMSOL comp1/geom1.getVertexCoord()\"");
      out.println("}");
    }
  }

  private static void validateVertices(double[][] vertices) {
    if (vertices.length != 3) {
      throw new RuntimeException(
          "Expected a three-dimensional geometry, got coordinate rows=" + vertices.length);
    }
    for (int dimension = 0; dimension < vertices.length; ++dimension) {
      if (vertices[dimension].length < POINT_IDS[POINT_IDS.length - 1]) {
        throw new RuntimeException(
            "Geometry has only "
                + vertices[dimension].length
                + " vertices; cannot read point entity 467.");
      }
    }
  }

  private static double scaleToMeters(String unit) {
    if (unit.equals("m")) {
      return 1.0;
    }
    if (unit.equals("cm")) {
      return 1.0e-2;
    }
    if (unit.equals("mm")) {
      return 1.0e-3;
    }
    if (unit.equals("um") || unit.equals("µm")) {
      return 1.0e-6;
    }
    throw new RuntimeException("Unsupported COMSOL geometry length unit: " + unit);
  }
}
