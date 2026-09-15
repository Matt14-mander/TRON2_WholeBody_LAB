#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

#include "tron2_ocs2/SolverCore.h"

namespace {

constexpr std::size_t kMaximumPendingBytes = 65536;

class FileDescriptor {
 public:
  explicit FileDescriptor(int value = -1) : value_(value) {}
  ~FileDescriptor() {
    if (value_ >= 0) ::close(value_);
  }
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  int get() const { return value_; }

 private:
  int value_;
};

template <typename Derived>
bool readEigen(std::istringstream& stream, Eigen::MatrixBase<Derived>& value) {
  for (Eigen::Index index = 0; index < value.size(); ++index) {
    if (!(stream >> value.derived().coeffRef(index))) return false;
  }
  return true;
}

std::string sanitizeError(std::string message) {
  for (char& character : message) {
    if (character == '\n' || character == '\r' || character == '\t') character = ' ';
  }
  return message;
}

void sendAll(int socket, const std::string& message) {
  std::size_t offset = 0;
  while (offset < message.size()) {
    const ssize_t sent = ::send(socket, message.data() + offset, message.size() - offset, MSG_NOSIGNAL);
    if (sent < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("send failed: ") + std::strerror(errno));
    }
    offset += static_cast<std::size_t>(sent);
  }
}

std::string solutionResponse(std::uint64_t requestId, const tron2_ocs2::Solution& solution) {
  std::ostringstream stream;
  stream << std::setprecision(17) << "OK " << requestId << ' ' << solution.time;
  for (const double value : solution.armPosition) stream << ' ' << value;
  for (const double value : solution.armVelocity) stream << ' ' << value;
  for (const double value : solution.armEffort) stream << ' ' << value;
  for (const double value : solution.baseVelocityCommand) stream << ' ' << value;
  for (Eigen::Index row = 0; row < solution.baseWrenchPrediction.rows(); ++row) {
    for (Eigen::Index column = 0; column < solution.baseWrenchPrediction.cols(); ++column) {
      stream << ' ' << solution.baseWrenchPrediction(row, column);
    }
  }
  stream << '\n';
  return stream.str();
}

std::string handleRequest(const std::string& line, tron2_ocs2::SolverCore& solver) {
  std::istringstream stream(line);
  std::string command;
  std::uint64_t requestId = 0;
  if (!(stream >> command >> requestId)) return "ERR 0 malformed_request\n";

  if (command == "RESET") {
    std::string trailing;
    if (stream >> trailing) return "ERR " + std::to_string(requestId) + " malformed_reset\n";
    solver.reset();
    return "OK_RESET " + std::to_string(requestId) + "\n";
  }
  if (command != "SOLVE") {
    return "ERR " + std::to_string(requestId) + " unknown_command\n";
  }

  tron2_ocs2::Observation observation;
  tron2_ocs2::EndEffectorTarget target;
  Eigen::Matrix<double, 4, 1> baseQuaternionWxyz;
  Eigen::Matrix<double, 4, 1> targetQuaternionWxyz;
  if (!(stream >> observation.time) || !readEigen(stream, observation.basePositionWorld) ||
      !readEigen(stream, baseQuaternionWxyz) || !readEigen(stream, observation.baseTwistBody) ||
      !readEigen(stream, observation.armPosition) || !readEigen(stream, observation.armVelocity) ||
      !readEigen(stream, target.positionWorld) || !readEigen(stream, targetQuaternionWxyz)) {
    return "ERR " + std::to_string(requestId) + " malformed_solve\n";
  }
  std::string trailing;
  if (stream >> trailing) return "ERR " + std::to_string(requestId) + " extra_fields\n";

  observation.baseOrientationWorld = Eigen::Quaterniond(
      baseQuaternionWxyz(0), baseQuaternionWxyz(1), baseQuaternionWxyz(2), baseQuaternionWxyz(3));
  target.orientationWorld = Eigen::Quaterniond(
      targetQuaternionWxyz(0), targetQuaternionWxyz(1), targetQuaternionWxyz(2), targetQuaternionWxyz(3));

  tron2_ocs2::Solution solution;
  std::string error;
  if (!solver.trySolve(observation, target, solution, &error)) {
    return "ERR " + std::to_string(requestId) + " " + sanitizeError(error) + "\n";
  }
  return solutionResponse(requestId, solution);
}

void serveClient(int client, tron2_ocs2::SolverCore& solver) {
  std::string pending;
  char buffer[4096];
  while (true) {
    const ssize_t received = ::recv(client, buffer, sizeof(buffer), 0);
    if (received == 0) return;
    if (received < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error(std::string("recv failed: ") + std::strerror(errno));
    }
    pending.append(buffer, static_cast<std::size_t>(received));
    if (pending.size() > kMaximumPendingBytes) {
      sendAll(client, "ERR 0 request_too_large\n");
      return;
    }
    std::size_t newline = 0;
    while ((newline = pending.find('\n')) != std::string::npos) {
      std::string line = pending.substr(0, newline);
      pending.erase(0, newline + 1);
      try {
        sendAll(client, handleRequest(line, solver));
      } catch (const std::exception& error) {
        sendAll(client, "ERR 0 " + sanitizeError(error.what()) + "\n");
      }
    }
  }
}

int parsePort(const char* text) {
  std::size_t consumed = 0;
  const int port = std::stoi(text, &consumed);
  if (text[consumed] != '\0' || port <= 0 || port > 65535) {
    throw std::invalid_argument("PORT must be in [1, 65535].");
  }
  return port;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5 || argc > 6) {
    std::cerr << "usage: tron2_ocs2_bridge TASK_INFO ROBOT_URDF GENERATED_LIBRARY_DIR PORT [BIND_ADDRESS]\n";
    return 2;
  }
  try {
    const int port = parsePort(argv[4]);
    const char* bindAddress = argc == 6 ? argv[5] : "127.0.0.1";
    tron2_ocs2::SolverCore solver(argv[1], argv[2], argv[3]);

    FileDescriptor server(::socket(AF_INET, SOCK_STREAM, 0));
    if (server.get() < 0) throw std::runtime_error(std::string("socket failed: ") + std::strerror(errno));
    int enabled = 1;
    ::setsockopt(server.get(), SOL_SOCKET, SO_REUSEADDR, &enabled, sizeof(enabled));

    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(static_cast<std::uint16_t>(port));
    if (::inet_pton(AF_INET, bindAddress, &address.sin_addr) != 1) {
      throw std::invalid_argument(std::string("Invalid IPv4 bind address: ") + bindAddress);
    }
    if (::bind(server.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) != 0) {
      throw std::runtime_error(std::string("bind failed: ") + std::strerror(errno));
    }
    if (::listen(server.get(), 1) != 0) {
      throw std::runtime_error(std::string("listen failed: ") + std::strerror(errno));
    }
    std::cout << "tron2_ocs2_bridge listening on " << bindAddress << ':' << port << std::endl;

    while (true) {
      sockaddr_in clientAddress{};
      socklen_t clientLength = sizeof(clientAddress);
      FileDescriptor client(::accept(server.get(), reinterpret_cast<sockaddr*>(&clientAddress), &clientLength));
      if (client.get() < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error(std::string("accept failed: ") + std::strerror(errno));
      }
      ::setsockopt(client.get(), IPPROTO_TCP, TCP_NODELAY, &enabled, sizeof(enabled));
      try {
        serveClient(client.get(), solver);
      } catch (const std::exception& error) {
        std::cerr << "client disconnected after error: " << error.what() << '\n';
      }
    }
  } catch (const std::exception& error) {
    std::cerr << "tron2_ocs2_bridge failed: " << error.what() << '\n';
    return 1;
  }
}
