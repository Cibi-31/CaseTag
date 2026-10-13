import { useState } from "react";
import { useNavigate } from "react-router-dom";

const API_URL = "http://127.0.0.1:8000";

function Login() {

  const navigate = useNavigate();

  const [role, setRole] = useState("common");

  const [email, setEmail] = useState("");

  const [password, setPassword] = useState("");

  const [error, setError] = useState("");

  const [loading, setLoading] = useState(false);


  const handleLogin = async (event) => {

    event.preventDefault();

    setError("");

    setLoading(true);

    try {

      const response = await fetch(
        `${API_URL}/api/auth/login`,
        {
          method: "POST",

          headers: {
            "Content-Type":
              "application/json"
          },

          body: JSON.stringify({
            email,
            password
          })
        }
      );

      const data =
        await response.json();

      if (!response.ok) {

        throw new Error(
          data.detail ||
          "Login failed."
        );
      }


      // Make sure selected role
      // matches authenticated role

      if (data.user.role !== role) {

        throw new Error(
          `This account is registered as ${data.user.role}.`
        );
      }


      localStorage.setItem(
        "casetag_token",
        data.access_token
      );


      localStorage.setItem(
        "casetag_user",
        JSON.stringify(
          data.user
        )
      );


      navigate("/");


    } catch (error) {

      setError(
        error.message
      );

    } finally {

      setLoading(false);

    }
  };


  return (

    <div className="auth-page">

      <div className="auth-card">

        <div className="auth-brand">

          <h1>CASETAG</h1>

          <span>
            LEGAL AI
          </span>

        </div>


        <h2>
          Welcome Back
        </h2>

        <p className="auth-subtitle">
          Choose your role to continue
        </p>


        <div className="role-selector">

          <button
            type="button"
            className={
              role === "lawyer"
                ? "role-card active"
                : "role-card"
            }
            onClick={() =>
              setRole("lawyer")
            }
          >

            <span className="role-icon">
              ⚖️
            </span>

            <strong>
              Lawyer
            </strong>

            <small>
              Professional Research
            </small>

          </button>


          <button
            type="button"
            className={
              role === "judge"
                ? "role-card active"
                : "role-card"
            }
            onClick={() =>
              setRole("judge")
            }
          >

            <span className="role-icon">
              👨‍⚖️
            </span>

            <strong>
              Judge
            </strong>

            <small>
              Judicial Research
            </small>

          </button>


          <button
            type="button"
            className={
              role === "common"
                ? "role-card active"
                : "role-card"
            }
            onClick={() =>
              setRole("common")
            }
          >

            <span className="role-icon">
              👤
            </span>

            <strong>
              Common Man
            </strong>

            <small>
              Understand My Case
            </small>

          </button>

        </div>


        <form
          onSubmit={handleLogin}
          className="auth-form"
        >

          <label>
            Email
          </label>

          <input
            type="email"
            value={email}
            onChange={(event) =>
              setEmail(
                event.target.value
              )
            }
            placeholder="Enter your email"
            required
          />


          <label>
            Password
          </label>

          <input
            type="password"
            value={password}
            onChange={(event) =>
              setPassword(
                event.target.value
              )
            }
            placeholder="Enter your password"
            required
          />


          {error && (

            <div className="auth-error">
              {error}
            </div>

          )}


          <button
            type="submit"
            className="auth-submit"
            disabled={loading}
          >

            {loading
              ? "Signing in..."
              : "Login →"}

          </button>

        </form>


        {role === "common" && (

          <p className="register-text">

            Don't have an account?

            {" "}

            <button
              type="button"
              onClick={() =>
                navigate("/register")
              }
            >
              Register
            </button>

          </p>

        )}

      </div>

    </div>
  );
}

export default Login;